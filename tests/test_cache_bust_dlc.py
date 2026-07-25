from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import struct
import tempfile
import unittest
import warnings
import zipfile
import zlib
from pathlib import Path
from unittest import mock

import scripts.cache_bust_dlc as cache_bust_module
from scripts.cache_bust_dlc import (
    EOCD_SIGNATURE,
    EOCD_STRUCT,
    MAX_ZIP_COMMENT_BYTES,
    ZIP64_EOCD_SIGNATURE,
    ZIP64_EOCD_STRUCT,
    ZIP64_LOCATOR_SIGNATURE,
    ZIP64_LOCATOR_STRUCT,
    _find_eocd,
    cache_bust_dlc,
    main,
)

REAL_DLC_ENV = "COSMOS_CACHE_BUST_REAL_DLC"
REAL_DLC_SOURCE_SHA256 = (
    "608a5558b8529f5055ef98d6e0ebeb6ff32c16fe0bc2ada9dcda8885dd06d0bc"
)
REAL_DLC_MARKER = "codex-cachebust-w8-p1-ar128-r9-20260725"
REAL_DLC_OUTPUT_SHA256 = (
    "98d6feb92afc1778570afbd2898018c345820bce3c2040f5f9c89ef8177673c1"
)


def _write_zip(
    path: Path,
    *,
    comment: bytes = b"",
    compression: int = zipfile.ZIP_DEFLATED,
    members: tuple[tuple[str, bytes], ...] = (
        ("graph/model.bin", b"ORIGINAL-PAYLOAD"),
        ("metadata.json", b'{"model": "fixture"}'),
    ),
) -> None:
    with zipfile.ZipFile(
        path, "w", compression=compression, allowZip64=True
    ) as archive:
        for name, content in members:
            archive.writestr(name, content)
        archive.comment = comment


def _eocd(path: Path):
    with path.open("rb") as handle:
        return _find_eocd(handle, path.stat().st_size)


def _member_contents(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path, "r") as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def _promote_to_genuine_zip64(path: Path) -> None:
    raw = path.read_bytes()
    eocd = _eocd(path)
    zip64_eocd_offset = eocd.offset
    zip64_eocd = ZIP64_EOCD_STRUCT.pack(
        ZIP64_EOCD_SIGNATURE,
        ZIP64_EOCD_STRUCT.size - 12,
        45,
        45,
        0,
        0,
        eocd.entries_on_disk,
        eocd.entry_count,
        eocd.central_directory_size,
        eocd.central_directory_offset,
    )
    locator = ZIP64_LOCATOR_STRUCT.pack(
        ZIP64_LOCATOR_SIGNATURE,
        0,
        zip64_eocd_offset,
        1,
    )
    standard_eocd = EOCD_STRUCT.pack(
        EOCD_SIGNATURE,
        0,
        0,
        0xFFFF,
        0xFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        len(eocd.comment),
    )
    path.write_bytes(
        raw[: eocd.offset]
        + zip64_eocd
        + locator
        + standard_eocd
        + eocd.comment
    )


def _replace_first_member_declared_crc(path: Path, crc32: int) -> None:
    raw = bytearray(path.read_bytes())
    local_header_at = raw.find(b"PK\x03\x04")
    central_header_at = raw.find(b"PK\x01\x02")
    if local_header_at < 0 or central_header_at < 0:
        raise AssertionError("fixture has no local or central ZIP header")
    struct.pack_into("<L", raw, local_header_at + 14, crc32)
    struct.pack_into("<L", raw, central_header_at + 16, crc32)
    path.write_bytes(raw)


def _flip_first_member_payload_byte(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        info = archive.infolist()[0]
    raw = bytearray(path.read_bytes())
    local_fields = struct.unpack_from("<4s5H3L2H", raw, info.header_offset)
    payload_at = (
        info.header_offset + 30 + local_fields[-2] + local_fields[-1]
    )
    if not info.compress_size:
        raise AssertionError("fixture member has no payload")
    raw[payload_at + info.compress_size // 2] ^= 0x01
    path.write_bytes(raw)


class CacheBustDlcTests(unittest.TestCase):
    def test_replaces_only_eocd_length_and_comment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.dlc"
            destination = root / "variant.dlc"
            _write_zip(
                source,
                comment=b"old-comment",
            )
            source_before = source.read_bytes()
            source_eocd = _eocd(source)
            marker = "aihub-ar128-first-r9-🤖"

            provenance = cache_bust_dlc(source, destination, marker)

            marker_bytes = marker.encode("utf-8")
            expected = (
                source_before[: source_eocd.immutable_prefix_length]
                + struct.pack("<H", len(marker_bytes))
                + marker_bytes
            )
            self.assertEqual(destination.read_bytes(), expected)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(
                _member_contents(destination), _member_contents(source)
            )
            self.assertNotEqual(
                hashlib.sha256(destination.read_bytes()).hexdigest(),
                hashlib.sha256(source_before).hexdigest(),
            )
            self.assertEqual(provenance["schema_version"], 1)
            self.assertEqual(
                provenance["operation"], "zip_eocd_comment_cache_bust"
            )
            self.assertEqual(provenance["source"]["name"], "source.dlc")
            self.assertEqual(provenance["output"]["name"], "variant.dlc")
            self.assertNotIn(str(root), json.dumps(provenance))
            self.assertTrue(
                provenance["invariants"][
                    "compressed_and_decoded_member_hashes_match"
                ]
            )
            self.assertEqual(provenance["invariants"]["member_count"], 2)
            self.assertEqual(len(provenance["members"]), 2)
            self.assertTrue(
                all(
                    len(member["raw_payload_sha256"]) == 64
                    and len(member["decoded_sha256"]) == 64
                    and member["declared_crc32_matches"]
                    for member in provenance["members"]
                )
            )

    def test_accepts_maximum_comment_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "variant.bin"
            _write_zip(source)
            marker = "x" * MAX_ZIP_COMMENT_BYTES

            provenance = cache_bust_dlc(source, destination, marker)

            self.assertEqual(
                provenance["output"]["comment_length_bytes"],
                MAX_ZIP_COMMENT_BYTES,
            )
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(archive.comment, marker.encode())

    def test_preserves_force_zip64_local_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.dlc"
            destination = root / "variant.dlc"
            with zipfile.ZipFile(
                source,
                "w",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as archive:
                with archive.open(
                    "model/context.bin", "w", force_zip64=True
                ) as member:
                    member.write(b"zip64-context-payload")
                archive.comment = b"original"
            source_contents = _member_contents(source)

            cache_bust_dlc(source, destination, "zip64-cache-key")

            self.assertEqual(_member_contents(destination), source_contents)
            self.assertEqual(
                _eocd(destination).comment, b"zip64-cache-key"
            )

    def test_preserves_genuine_zip64_eocd_and_locator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.dlc"
            destination = root / "variant.dlc"
            _write_zip(source, comment=b"original")
            _promote_to_genuine_zip64(source)
            source_eocd = _eocd(source)
            self.assertTrue(source_eocd.uses_zip64)
            self.assertEqual(source_eocd.effective_entry_count, 2)
            source_before = source.read_bytes()
            source_contents = _member_contents(source)

            provenance = cache_bust_dlc(
                source, destination, "zip64-cache-key"
            )

            destination_eocd = _eocd(destination)
            self.assertTrue(destination_eocd.uses_zip64)
            marker_bytes = b"zip64-cache-key"
            self.assertEqual(
                destination.read_bytes(),
                source_before[: source_eocd.immutable_prefix_length]
                + struct.pack("<H", len(marker_bytes))
                + marker_bytes,
            )
            self.assertEqual(_member_contents(destination), source_contents)
            self.assertTrue(
                provenance["invariants"]["zip64_eocd_present"]
            )
            self.assertTrue(
                provenance["invariants"][
                    "zip64_structures_validated_when_present"
                ]
            )

    def test_refuses_missing_and_malformed_zip64_structures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            missing = root / "missing-locator.dlc"
            _write_zip(missing, comment=b"original")
            missing_eocd = _eocd(missing)
            missing_raw = missing.read_bytes()
            sentinel_eocd = EOCD_STRUCT.pack(
                EOCD_SIGNATURE,
                0,
                0,
                0xFFFF,
                0xFFFF,
                0xFFFFFFFF,
                0xFFFFFFFF,
                len(missing_eocd.comment),
            )
            missing.write_bytes(
                missing_raw[: missing_eocd.offset]
                + sentinel_eocd
                + missing_eocd.comment
            )
            with self.assertRaisesRegex(ValueError, "require.*locator"):
                cache_bust_dlc(
                    missing, root / "missing-output.dlc", "marker"
                )

            bad_locator = root / "bad-locator.dlc"
            _write_zip(bad_locator, comment=b"original")
            _promote_to_genuine_zip64(bad_locator)
            bad_locator_raw = bytearray(bad_locator.read_bytes())
            locator_at = bad_locator_raw.rfind(ZIP64_LOCATOR_SIGNATURE)
            self.assertGreaterEqual(locator_at, 0)
            struct.pack_into("<Q", bad_locator_raw, locator_at + 8, 1)
            bad_locator.write_bytes(bad_locator_raw)
            with self.assertRaisesRegex(ValueError, "invalid signature"):
                cache_bust_dlc(
                    bad_locator, root / "bad-locator-output.dlc", "marker"
                )

            bad_record = root / "bad-record.dlc"
            _write_zip(bad_record, comment=b"original")
            _promote_to_genuine_zip64(bad_record)
            bad_record_raw = bytearray(bad_record.read_bytes())
            record_at = bad_record_raw.rfind(ZIP64_EOCD_SIGNATURE)
            self.assertGreaterEqual(record_at, 0)
            struct.pack_into("<Q", bad_record_raw, record_at + 4, 43)
            bad_record.write_bytes(bad_record_raw)
            with self.assertRaisesRegex(ValueError, "below 44 bytes"):
                cache_bust_dlc(
                    bad_record, root / "bad-record-output.dlc", "marker"
                )

    def test_rejects_marker_over_byte_limit_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "variant.bin"
            _write_zip(source)
            source_before = source.read_bytes()
            marker = "é" * ((MAX_ZIP_COMMENT_BYTES // 2) + 1)

            with self.assertRaisesRegex(ValueError, "too large"):
                cache_bust_dlc(source, destination, marker)

            self.assertEqual(source.read_bytes(), source_before)
            self.assertFalse(destination.exists())

    def test_rejects_comment_that_would_not_change_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "variant.bin"
            _write_zip(source, comment=b"same")

            with self.assertRaisesRegex(ValueError, "already"):
                cache_bust_dlc(source, destination, "same")

            self.assertFalse(destination.exists())

    def test_refuses_existing_destination_without_modifying_either_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "variant.bin"
            _write_zip(source)
            destination.write_bytes(b"keep-me")
            source_before = source.read_bytes()

            with self.assertRaises(FileExistsError):
                cache_bust_dlc(source, destination, "new-marker")

            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(destination.read_bytes(), b"keep-me")

    def test_refuses_source_as_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.bin"
            _write_zip(source)
            source_before = source.read_bytes()

            with self.assertRaises((FileExistsError, ValueError)):
                cache_bust_dlc(source, source, "new-marker")

            self.assertEqual(source.read_bytes(), source_before)

    def test_atomic_publish_failure_leaves_no_output_or_temporary_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "variant.bin"
            _write_zip(source)
            source_before = source.read_bytes()

            with mock.patch.object(
                cache_bust_module.os,
                "link",
                side_effect=OSError("fixture failure"),
            ):
                with self.assertRaisesRegex(OSError, "atomically publish"):
                    cache_bust_dlc(source, destination, "new-marker")

            self.assertEqual(source.read_bytes(), source_before)
            self.assertFalse(destination.exists())
            self.assertEqual(
                list(root.glob(f".{destination.name}.*.tmp")), []
            )

    def test_post_commit_cleanup_failure_keeps_valid_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "variant.bin"
            _write_zip(source)

            with mock.patch.object(
                cache_bust_module,
                "_cleanup_published_temporary",
                return_value=False,
            ):
                provenance = cache_bust_dlc(
                    source, destination, "new-marker"
                )

            self.assertEqual(
                _member_contents(destination), _member_contents(source)
            )
            self.assertFalse(
                provenance["invariants"]["temporary_cleanup_succeeded"]
            )

    def test_temporary_output_is_never_reopened_by_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "variant.bin"
            _write_zip(source)
            original_path_open = Path.open

            def guarded_open(path: Path, *args, **kwargs):
                if path.name.startswith(f".{destination.name}."):
                    raise AssertionError("temporary path was reopened")
                return original_path_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", guarded_open):
                cache_bust_dlc(source, destination, "new-marker")

            self.assertEqual(
                _member_contents(destination), _member_contents(source)
            )

    @unittest.skipIf(
        os.name == "nt",
        "Windows does not permit replacing this open temporary file",
    )
    def test_refuses_temporary_path_symlink_swap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "variant.bin"
            victim = root / "victim.bin"
            _write_zip(source)
            victim.write_bytes(b"do-not-touch")
            source_before = source.read_bytes()
            created: dict[str, Path] = {}
            original_mkstemp = tempfile.mkstemp
            original_write_variant = cache_bust_module._write_variant

            def recording_mkstemp(*args, **kwargs):
                descriptor, name = original_mkstemp(*args, **kwargs)
                created["path"] = Path(name)
                return descriptor, name

            def swapping_write_variant(*args, **kwargs):
                result = original_write_variant(*args, **kwargs)
                temporary = created["path"]
                temporary.unlink()
                temporary.symlink_to(victim)
                return result

            with (
                mock.patch.object(
                    cache_bust_module.tempfile,
                    "mkstemp",
                    recording_mkstemp,
                ),
                mock.patch.object(
                    cache_bust_module,
                    "_write_variant",
                    swapping_write_variant,
                ),
            ):
                with self.assertRaisesRegex(
                    OSError, "Temporary output path changed"
                ):
                    cache_bust_dlc(source, destination, "new-marker")

            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(victim.read_bytes(), b"do-not-touch")
            self.assertFalse(destination.exists())

    def test_refuses_trailing_bytes_and_corrupt_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            trailing = root / "trailing.bin"
            _write_zip(trailing)
            with trailing.open("ab") as handle:
                handle.write(b"trailing-junk")
            with self.assertRaisesRegex(ValueError, "terminal ZIP EOCD"):
                cache_bust_dlc(
                    trailing, root / "trailing-output.bin", "marker"
                )

            corrupt = root / "corrupt.bin"
            _write_zip(
                corrupt,
                compression=zipfile.ZIP_DEFLATED,
                members=(("model.bin", b"ORIGINAL-PAYLOAD"),),
            )
            _flip_first_member_payload_byte(corrupt)
            with self.assertRaisesRegex(ValueError, "Invalid or unsupported"):
                cache_bust_dlc(
                    corrupt, root / "corrupt-output.bin", "marker"
                )

    def test_tolerates_bad_declared_crc_only_for_stored_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"QAIRT-STORED-PAYLOAD"
            source = root / "source.dlc"
            destination = root / "variant.dlc"
            _write_zip(
                source,
                compression=zipfile.ZIP_STORED,
                members=(("dlc.metadata2.1.0", payload),),
            )
            _replace_first_member_declared_crc(source, 0)
            source_before = source.read_bytes()
            with zipfile.ZipFile(source) as archive:
                with self.assertRaises(zipfile.BadZipFile):
                    archive.read("dlc.metadata2.1.0")

            provenance = cache_bust_dlc(
                source, destination, "qairt-cache-bust"
            )

            self.assertEqual(source.read_bytes(), source_before)
            member = provenance["members"][0]
            actual_crc32 = zlib.crc32(payload) & 0xFFFFFFFF
            self.assertEqual(member["declared_crc32"], "00000000")
            self.assertEqual(
                member["actual_crc32"], f"{actual_crc32:08x}"
            )
            self.assertFalse(member["declared_crc32_matches"])
            self.assertEqual(
                member["raw_payload_sha256"],
                hashlib.sha256(payload).hexdigest(),
            )
            self.assertEqual(
                member["decoded_sha256"],
                hashlib.sha256(payload).hexdigest(),
            )
            self.assertEqual(
                provenance["invariants"][
                    "stored_declared_crc_mismatch_members"
                ],
                ["dlc.metadata2.1.0"],
            )

            compressed = root / "compressed.dlc"
            _write_zip(
                compressed,
                compression=zipfile.ZIP_DEFLATED,
                members=(("model.bin", payload),),
            )
            _replace_first_member_declared_crc(compressed, 0)
            with self.assertRaisesRegex(
                ValueError, "Invalid or unsupported"
            ):
                cache_bust_dlc(
                    compressed,
                    root / "compressed-output.dlc",
                    "marker",
                )

    @unittest.skipUnless(
        os.environ.get(REAL_DLC_ENV),
        f"set {REAL_DLC_ENV} to run the real-DLC integration",
    )
    def test_real_qairt_dlc_reproduces_r9_artifact_hash(self) -> None:
        source = Path(os.environ[REAL_DLC_ENV])
        source_before = source.stat()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cache-bust-r9.dlc"

            provenance = cache_bust_dlc(
                source,
                output,
                REAL_DLC_MARKER,
            )

            self.assertEqual(
                provenance["source"]["sha256"], REAL_DLC_SOURCE_SHA256
            )
            self.assertEqual(
                provenance["output"]["sha256"], REAL_DLC_OUTPUT_SHA256
            )
            self.assertEqual(
                hashlib.sha256(output.read_bytes()).hexdigest(),
                REAL_DLC_OUTPUT_SHA256,
            )
            mismatches = [
                member
                for member in provenance["members"]
                if not member["declared_crc32_matches"]
            ]
            self.assertEqual(
                [member["name"] for member in mismatches],
                [
                    "dlc.metadata2.1.0",
                    "model",
                    "model.params",
                    "model.params.bin",
                ],
            )
            self.assertTrue(
                all(
                    member["compression"] == zipfile.ZIP_STORED
                    and member["raw_payload_sha256"]
                    == member["decoded_sha256"]
                    for member in mismatches
                )
            )
        source_after = source.stat()
        self.assertEqual(
            (
                source_before.st_dev,
                source_before.st_ino,
                source_before.st_size,
                source_before.st_mtime_ns,
            ),
            (
                source_after.st_dev,
                source_after.st_ino,
                source_after.st_size,
                source_after.st_mtime_ns,
            ),
        )

    def test_refuses_ambiguous_terminal_eocd_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ambiguous.bin"
            suffix = b"fake-record-tail"
            fake_eocd = EOCD_STRUCT.pack(
                EOCD_SIGNATURE,
                0,
                0,
                0,
                0,
                0,
                0,
                len(suffix),
            )
            _write_zip(source, comment=fake_eocd + suffix)

            with self.assertRaisesRegex(ValueError, "multiple plausible"):
                cache_bust_dlc(
                    source, root / "ambiguous-output.bin", "marker"
                )

    def test_refuses_multidisk_duplicate_and_unsafe_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            multidisk = root / "multidisk.bin"
            _write_zip(multidisk)
            raw = bytearray(multidisk.read_bytes())
            eocd = _eocd(multidisk)
            struct.pack_into("<H", raw, eocd.offset + 4, 1)
            multidisk.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "Multi-disk"):
                cache_bust_dlc(
                    multidisk, root / "multidisk-output.bin", "marker"
                )

            duplicate = root / "duplicate.bin"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                _write_zip(
                    duplicate,
                    members=(
                        ("model.bin", b"first"),
                        ("model.bin", b"second"),
                    ),
                )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                cache_bust_dlc(
                    duplicate, root / "duplicate-output.bin", "marker"
                )

            unsafe = root / "unsafe.bin"
            _write_zip(
                unsafe,
                members=(("../outside.bin", b"payload"),),
            )
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                cache_bust_dlc(
                    unsafe, root / "unsafe-output.bin", "marker"
                )

    def test_cli_emits_machine_readable_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "variant.bin"
            _write_zip(source)
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                result = main(
                    [
                        str(source),
                        str(destination),
                        "--marker",
                        "tutorial-cache-key-r1",
                    ]
                )

            self.assertEqual(result, 0)
            provenance = json.loads(output.getvalue())
            self.assertEqual(
                provenance["output"]["sha256"],
                hashlib.sha256(destination.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                provenance["output"]["comment_utf8"],
                "tutorial-cache-key-r1",
            )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_refuses_existing_destination_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            target = root / "target.bin"
            destination = root / "variant.bin"
            _write_zip(source)
            target.write_bytes(b"keep-me")
            try:
                destination.symlink_to(target)
            except OSError:
                self.skipTest("creating symlinks is not permitted")

            with self.assertRaises(FileExistsError):
                cache_bust_dlc(source, destination, "marker")

            self.assertEqual(target.read_bytes(), b"keep-me")


if __name__ == "__main__":
    unittest.main()
