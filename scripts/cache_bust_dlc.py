#!/usr/bin/env python3
"""Create a byte-preserving AI Hub DLC variant using a ZIP comment.

QAIRT DLC files are ZIP containers.  Changing the end-of-central-directory
(EOCD) archive comment gives AI Hub a distinct artifact hash without rewriting
the central directory, local headers, or member payloads.

This helper deliberately does not use ``ZipFile`` to write the destination:
Python's ZIP writer is allowed to rewrite member metadata and compressed
payloads.  Instead, it copies every byte before the EOCD comment-length field,
replaces that two-byte field, and appends the requested UTF-8 comment.  It then
validates the completed temporary archive before atomically publishing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import struct
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Sequence

EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
LOCAL_FILE_SIGNATURE = b"PK\x03\x04"
EOCD_STRUCT = struct.Struct("<4s4H2LH")
ZIP64_EOCD_STRUCT = struct.Struct("<4sQ2H2L4Q")
ZIP64_LOCATOR_STRUCT = struct.Struct("<4sLQL")
LOCAL_FILE_STRUCT = struct.Struct("<4s5H3L2H")
EOCD_FIXED_SIZE = EOCD_STRUCT.size
ZIP64_EOCD_MIN_RECORD_DATA_SIZE = ZIP64_EOCD_STRUCT.size - 12
MAX_ZIP_COMMENT_BYTES = (1 << 16) - 1
EOCD_SEARCH_BYTES = EOCD_FIXED_SIZE + MAX_ZIP_COMMENT_BYTES
COPY_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class Eocd:
    """Validated location and fields of a ZIP EOCD record."""

    offset: int
    disk_number: int
    central_directory_disk: int
    entries_on_disk: int
    entry_count: int
    central_directory_size: int
    central_directory_offset: int
    comment: bytes
    effective_entry_count: int
    effective_central_directory_size: int
    effective_central_directory_offset: int
    uses_zip64: bool
    zip64_eocd_offset: int | None

    @property
    def immutable_prefix_length(self) -> int:
        """Number of bytes before the EOCD comment-length field."""

        return self.offset + EOCD_FIXED_SIZE - 2


def _find_eocd(handle: BinaryIO, file_size: int) -> Eocd:
    if file_size < EOCD_FIXED_SIZE:
        raise ValueError("Source is too small to contain a ZIP EOCD record")

    search_start = max(0, file_size - EOCD_SEARCH_BYTES)
    handle.seek(search_start)
    tail = handle.read(file_size - search_start)
    candidates: list[Eocd] = []
    search_at = 0
    while True:
        relative_offset = tail.find(EOCD_SIGNATURE, search_at)
        if relative_offset < 0:
            break
        search_at = relative_offset + 1
        if relative_offset + EOCD_FIXED_SIZE > len(tail):
            continue
        fields = EOCD_STRUCT.unpack_from(tail, relative_offset)
        comment_length = fields[-1]
        absolute_offset = search_start + relative_offset
        if absolute_offset + EOCD_FIXED_SIZE + comment_length != file_size:
            continue
        candidates.append(
            Eocd(
                offset=absolute_offset,
                disk_number=fields[1],
                central_directory_disk=fields[2],
                entries_on_disk=fields[3],
                entry_count=fields[4],
                central_directory_size=fields[5],
                central_directory_offset=fields[6],
                comment=tail[
                    relative_offset
                    + EOCD_FIXED_SIZE : relative_offset
                    + EOCD_FIXED_SIZE
                    + comment_length
                ],
                effective_entry_count=fields[4],
                effective_central_directory_size=fields[5],
                effective_central_directory_offset=fields[6],
                uses_zip64=False,
                zip64_eocd_offset=None,
            )
        )

    if not candidates:
        raise ValueError(
            "Source has no unambiguous terminal ZIP EOCD record "
            "(trailing bytes and truncated comments are refused)"
        )
    if len(candidates) != 1:
        raise ValueError(
            "Source has multiple plausible terminal ZIP EOCD records; "
            "refusing an ambiguous archive"
        )

    eocd = candidates[0]
    if eocd.disk_number != 0 or eocd.central_directory_disk != 0:
        raise ValueError("Multi-disk ZIP archives are not supported")
    return _resolve_zip64_eocd(handle, eocd)


def _read_exact_at(handle: BinaryIO, offset: int, size: int) -> bytes:
    if offset < 0:
        return b""
    handle.seek(offset)
    return handle.read(size)


def _resolve_zip64_eocd(handle: BinaryIO, eocd: Eocd) -> Eocd:
    needs_zip64 = (
        eocd.entries_on_disk == 0xFFFF
        or eocd.entry_count == 0xFFFF
        or eocd.central_directory_size == 0xFFFFFFFF
        or eocd.central_directory_offset == 0xFFFFFFFF
    )
    locator_offset = eocd.offset - ZIP64_LOCATOR_STRUCT.size
    locator_bytes = _read_exact_at(
        handle, locator_offset, ZIP64_LOCATOR_STRUCT.size
    )
    has_locator = locator_bytes.startswith(ZIP64_LOCATOR_SIGNATURE)
    if needs_zip64 and not has_locator:
        raise ValueError(
            "ZIP64 sentinel values require a ZIP64 EOCD locator immediately "
            "before the standard EOCD"
        )
    if not has_locator:
        if (
            eocd.entries_on_disk != eocd.entry_count
            or eocd.central_directory_offset
            + eocd.central_directory_size
            > eocd.offset
        ):
            raise ValueError("EOCD central-directory fields are inconsistent")
        return eocd

    (
        _,
        zip64_eocd_disk,
        zip64_eocd_offset,
        total_disks,
    ) = ZIP64_LOCATOR_STRUCT.unpack(locator_bytes)
    if zip64_eocd_disk != 0 or total_disks != 1:
        raise ValueError("Multi-disk ZIP64 archives are not supported")
    zip64_fixed = _read_exact_at(
        handle, zip64_eocd_offset, ZIP64_EOCD_STRUCT.size
    )
    if len(zip64_fixed) != ZIP64_EOCD_STRUCT.size:
        raise ValueError("ZIP64 EOCD record is missing or truncated")
    zip64_fields = ZIP64_EOCD_STRUCT.unpack(zip64_fixed)
    if zip64_fields[0] != ZIP64_EOCD_SIGNATURE:
        raise ValueError("ZIP64 EOCD locator points to an invalid signature")

    zip64_record_data_size = zip64_fields[1]
    if zip64_record_data_size < ZIP64_EOCD_MIN_RECORD_DATA_SIZE:
        raise ValueError("ZIP64 EOCD record-data size is below 44 bytes")
    zip64_record_end = (
        zip64_eocd_offset + 12 + zip64_record_data_size
    )
    if zip64_record_end != locator_offset:
        raise ValueError(
            "ZIP64 EOCD record does not end immediately before its locator"
        )

    zip64_disk_number = zip64_fields[4]
    zip64_central_directory_disk = zip64_fields[5]
    zip64_entries_on_disk = zip64_fields[6]
    zip64_entry_count = zip64_fields[7]
    zip64_central_directory_size = zip64_fields[8]
    zip64_central_directory_offset = zip64_fields[9]
    if zip64_disk_number != 0 or zip64_central_directory_disk != 0:
        raise ValueError("Multi-disk ZIP64 archives are not supported")
    if zip64_entries_on_disk != zip64_entry_count:
        raise ValueError(
            "ZIP64 EOCD has inconsistent per-disk and total entry counts"
        )
    if (
        zip64_central_directory_offset + zip64_central_directory_size
        > zip64_eocd_offset
    ):
        raise ValueError(
            "ZIP64 central-directory range extends beyond the ZIP64 EOCD"
        )

    standard_and_zip64 = (
        (
            eocd.entries_on_disk,
            0xFFFF,
            zip64_entries_on_disk,
            "entries-on-disk",
        ),
        (eocd.entry_count, 0xFFFF, zip64_entry_count, "entry-count"),
        (
            eocd.central_directory_size,
            0xFFFFFFFF,
            zip64_central_directory_size,
            "central-directory-size",
        ),
        (
            eocd.central_directory_offset,
            0xFFFFFFFF,
            zip64_central_directory_offset,
            "central-directory-offset",
        ),
    )
    for standard, sentinel, zip64_value, label in standard_and_zip64:
        if standard != sentinel and standard != zip64_value:
            raise ValueError(
                f"Standard EOCD and ZIP64 EOCD disagree on {label}"
            )

    return Eocd(
        offset=eocd.offset,
        disk_number=eocd.disk_number,
        central_directory_disk=eocd.central_directory_disk,
        entries_on_disk=eocd.entries_on_disk,
        entry_count=eocd.entry_count,
        central_directory_size=eocd.central_directory_size,
        central_directory_offset=eocd.central_directory_offset,
        comment=eocd.comment,
        effective_entry_count=zip64_entry_count,
        effective_central_directory_size=zip64_central_directory_size,
        effective_central_directory_offset=zip64_central_directory_offset,
        uses_zip64=True,
        zip64_eocd_offset=zip64_eocd_offset,
    )


def _member_name_is_safe(name: str) -> bool:
    if not name or "\x00" in name:
        return False
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return False
    first_component = normalized.split("/", 1)[0]
    if len(first_component) >= 2 and first_component[1] == ":":
        return False
    return ".." not in PurePosixPath(normalized).parts


def _hash_stream(handle: BinaryIO, size: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    crc32 = 0
    remaining = size
    while remaining:
        chunk = handle.read(min(COPY_CHUNK_BYTES, remaining))
        if not chunk:
            raise ValueError("Archive ended inside a member payload")
        digest.update(chunk)
        crc32 = zlib.crc32(chunk, crc32)
        remaining -= len(chunk)
    return digest.hexdigest(), crc32 & 0xFFFFFFFF


def _validate_archive(
    handle: BinaryIO,
    eocd: Eocd,
    *,
    label: str,
) -> list[dict[str, Any]]:
    """Validate a ZIP and return hashes of compressed and decoded members."""

    try:
        handle.seek(0)
        with zipfile.ZipFile(handle, "r") as archive:
            if archive.comment != eocd.comment:
                raise ValueError("ZIP parser and EOCD scanner disagree on comment")
            infos = archive.infolist()
            if not infos:
                raise ValueError("Empty ZIP archives are not accepted as DLCs")
            if len(infos) != eocd.effective_entry_count:
                raise ValueError(
                    "ZIP member count does not match the EOCD entry count"
                )

            names: set[str] = set()
            spans: list[tuple[int, int, str]] = []
            manifest: list[dict[str, Any]] = []
            for index, info in enumerate(infos):
                original_name = info.orig_filename
                if original_name != info.filename or not _member_name_is_safe(
                    original_name
                ):
                    raise ValueError(
                        f"Unsafe or ambiguous ZIP member name: {original_name!r}"
                    )
                if info.filename in names:
                    raise ValueError(
                        f"Duplicate ZIP member name: {info.filename!r}"
                    )
                names.add(info.filename)
                if info.flag_bits & 0x1:
                    raise ValueError(
                        f"Encrypted ZIP member is unsupported: {info.filename!r}"
                    )
                if info.flag_bits & 0x60:
                    raise ValueError(
                        "Patched-data or strong-encryption ZIP member is "
                        f"unsupported: {info.filename!r}"
                    )
                if info.header_offset < 0:
                    raise ValueError(
                        f"Negative local-header offset: {info.filename!r}"
                    )

                handle.seek(info.header_offset)
                local_header = handle.read(LOCAL_FILE_STRUCT.size)
                if len(local_header) != LOCAL_FILE_STRUCT.size:
                    raise ValueError(
                        f"Truncated local header: {info.filename!r}"
                    )
                local_fields = LOCAL_FILE_STRUCT.unpack(local_header)
                if local_fields[0] != LOCAL_FILE_SIGNATURE:
                    raise ValueError(
                        f"Invalid local-header signature: {info.filename!r}"
                    )
                local_flag_bits = local_fields[2]
                local_compression = local_fields[3]
                if local_flag_bits != info.flag_bits:
                    raise ValueError(
                        f"Local and central flags disagree: {info.filename!r}"
                    )
                if local_compression != info.compress_type:
                    raise ValueError(
                        "Local and central compression methods disagree: "
                        f"{info.filename!r}"
                    )
                filename_length = local_fields[-2]
                extra_length = local_fields[-1]
                local_filename_bytes = handle.read(filename_length)
                if len(local_filename_bytes) != filename_length:
                    raise ValueError(
                        f"Truncated local filename: {info.filename!r}"
                    )
                filename_encoding = (
                    "utf-8" if local_flag_bits & 0x800 else "cp437"
                )
                local_filename = local_filename_bytes.decode(
                    filename_encoding
                )
                if local_filename != info.orig_filename:
                    raise ValueError(
                        "Local and central filenames disagree: "
                        f"{info.filename!r}"
                    )
                payload_offset = (
                    info.header_offset
                    + LOCAL_FILE_STRUCT.size
                    + filename_length
                    + extra_length
                )
                payload_end = payload_offset + info.compress_size
                if payload_end > archive.start_dir:
                    raise ValueError(
                        f"Member payload extends beyond central directory: "
                        f"{info.filename!r}"
                    )
                spans.append((info.header_offset, payload_end, info.filename))

                handle.seek(payload_offset)
                raw_payload_sha256, actual_crc32 = _hash_stream(
                    handle, info.compress_size
                )
                if info.compress_type == zipfile.ZIP_STORED:
                    if info.compress_size != info.file_size:
                        raise ValueError(
                            "Stored member has different compressed and "
                            f"decoded sizes: {info.filename!r}"
                        )
                    decoded_sha256 = raw_payload_sha256
                    decoded_size = info.compress_size
                else:
                    decoded_digest = hashlib.sha256()
                    decoded_crc32 = 0
                    decoded_size = 0
                    with archive.open(info, "r") as member:
                        for chunk in iter(
                            lambda: member.read(COPY_CHUNK_BYTES), b""
                        ):
                            decoded_digest.update(chunk)
                            decoded_crc32 = zlib.crc32(
                                chunk, decoded_crc32
                            )
                            decoded_size += len(chunk)
                    decoded_sha256 = decoded_digest.hexdigest()
                    actual_crc32 = decoded_crc32 & 0xFFFFFFFF
                if decoded_size != info.file_size:
                    raise ValueError(
                        f"Decoded size mismatch for {info.filename!r}"
                    )
                manifest.append(
                    {
                        "index": index,
                        "name": info.filename,
                        "crc32": f"{info.CRC:08x}",
                        "declared_crc32": f"{info.CRC:08x}",
                        "actual_crc32": f"{actual_crc32:08x}",
                        "declared_crc32_matches": info.CRC == actual_crc32,
                        "compression": info.compress_type,
                        "compressed_size_bytes": info.compress_size,
                        "decoded_size_bytes": info.file_size,
                        "raw_payload_sha256": raw_payload_sha256,
                        "decoded_sha256": decoded_sha256,
                    }
                )

            spans.sort()
            for previous, current in zip(spans, spans[1:]):
                if previous[1] > current[0]:
                    raise ValueError(
                        "Overlapping ZIP members are refused: "
                        f"{previous[2]!r} and {current[2]!r}"
                    )
            return manifest
    except (OSError, RuntimeError, NotImplementedError, zipfile.BadZipFile) as exc:
        raise ValueError(
            f"Invalid or unsupported ZIP/DLC archive: {label}"
        ) from exc


def _manifest_sha256(manifest: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _file_identity(stat_result: os.stat_result) -> tuple[int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )


def _inode_identity(stat_result: os.stat_result) -> tuple[int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
    )


def _sha256_handle(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(COPY_CHUNK_BYTES), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _write_variant(
    source_handle: BinaryIO,
    output_handle: BinaryIO,
    *,
    immutable_prefix_length: int,
    marker_bytes: bytes,
) -> tuple[str, str, str]:
    """Write the variant and return source, output, and prefix SHA-256."""

    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    prefix_digest = hashlib.sha256()
    copied_prefix = 0
    source_handle.seek(0)
    output_handle.seek(0)
    output_handle.truncate()
    while True:
        chunk = source_handle.read(COPY_CHUNK_BYTES)
        if not chunk:
            break
        source_digest.update(chunk)
        if copied_prefix < immutable_prefix_length:
            prefix_chunk = chunk[
                : min(len(chunk), immutable_prefix_length - copied_prefix)
            ]
            output_handle.write(prefix_chunk)
            output_digest.update(prefix_chunk)
            prefix_digest.update(prefix_chunk)
            copied_prefix += len(prefix_chunk)
    if copied_prefix != immutable_prefix_length:
        raise ValueError("Source changed or ended while copying EOCD prefix")

    replacement = struct.pack("<H", len(marker_bytes)) + marker_bytes
    output_handle.write(replacement)
    output_digest.update(replacement)
    output_handle.flush()
    os.fsync(output_handle.fileno())
    return (
        source_digest.hexdigest(),
        output_digest.hexdigest(),
        prefix_digest.hexdigest(),
    )


def _cleanup_published_temporary(temporary: Path) -> bool:
    try:
        temporary.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _publish_new_file(
    temporary: Path,
    destination: Path,
    *,
    expected_identity: tuple[int, int, int],
) -> None:
    """Atomically publish without ever replacing an existing path."""

    try:
        os.link(
            temporary,
            destination,
            follow_symlinks=False,
        )
    except FileExistsError:
        raise FileExistsError(
            f"Destination already exists; refusing to replace it: {destination}"
        ) from None
    except OSError as exc:
        raise OSError(
            "Cannot atomically publish with a same-directory hard link; "
            f"destination was not created: {destination}"
        ) from exc

    published_stat = destination.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(published_stat.st_mode)
        or _inode_identity(published_stat) != expected_identity
    ):
        try:
            destination.unlink()
        finally:
            _cleanup_published_temporary(temporary)
        raise OSError("Published destination identity failed verification")


def cache_bust_dlc(
    source: str | Path,
    destination: str | Path,
    marker: str,
) -> dict[str, Any]:
    """Create a validated DLC variant and return JSON-ready provenance.

    The destination must not exist.  The source is opened read-only and is
    never renamed, replaced, truncated, or otherwise modified.
    """

    source_path = Path(source)
    destination_path = Path(destination)
    if not isinstance(marker, str):
        raise TypeError("marker must be a Python str encoded as UTF-8")
    marker_bytes = marker.encode("utf-8")
    if not marker_bytes:
        raise ValueError("marker must not be empty")
    if len(marker_bytes) > MAX_ZIP_COMMENT_BYTES:
        raise ValueError(
            "UTF-8 marker is too large for a ZIP comment: "
            f"{len(marker_bytes)} > {MAX_ZIP_COMMENT_BYTES} bytes"
        )
    if not source_path.is_file():
        raise ValueError(f"Source is not a regular file: {source_path}")
    if not destination_path.parent.is_dir():
        raise ValueError(
            f"Destination parent directory does not exist: "
            f"{destination_path.parent}"
        )
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError(
            f"Destination already exists; refusing to replace it: "
            f"{destination_path}"
        )
    if source_path.resolve(strict=True) == destination_path.resolve(
        strict=False
    ):
        raise ValueError("Source and destination must be different paths")

    with source_path.open("rb") as source_handle:
        source_before = os.fstat(source_handle.fileno())
        if not stat.S_ISREG(source_before.st_mode):
            raise ValueError(f"Source is not a regular file: {source_path}")
        eocd = _find_eocd(source_handle, source_before.st_size)
        if eocd.comment == marker_bytes:
            raise ValueError(
                "Requested marker is already the archive comment; "
                "it would not produce a new artifact hash"
            )
        source_manifest = _validate_archive(
            source_handle,
            eocd,
            label=source_path.name,
        )

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            dir=destination_path.parent,
        )
        temporary_path = Path(temporary_name)
        published = False
        temporary_cleanup_succeeded = True
        try:
            with os.fdopen(file_descriptor, "w+b") as output_handle:
                (
                    source_sha256,
                    output_sha256,
                    immutable_prefix_sha256,
                ) = _write_variant(
                    source_handle,
                    output_handle,
                    immutable_prefix_length=eocd.immutable_prefix_length,
                    marker_bytes=marker_bytes,
                )
                source_after = os.fstat(source_handle.fileno())
                if _file_identity(source_before) != _file_identity(
                    source_after
                ):
                    raise ValueError(
                        "Source changed while the cache-bust copy was made"
                    )
                if output_sha256 == source_sha256:
                    raise ValueError(
                        "Cache-bust variant did not produce a new SHA-256"
                    )

                output_handle.seek(0, os.SEEK_END)
                output_size = output_handle.tell()
                output_handle.seek(0)
                output_eocd = _find_eocd(output_handle, output_size)
                if output_eocd.comment != marker_bytes:
                    raise ValueError(
                        "Output ZIP comment does not match requested marker"
                    )
                if (
                    output_eocd.immutable_prefix_length
                    != eocd.immutable_prefix_length
                ):
                    raise ValueError("Output EOCD moved unexpectedly")
                output_manifest = _validate_archive(
                    output_handle,
                    output_eocd,
                    label=destination_path.name,
                )
                if output_manifest != source_manifest:
                    raise ValueError(
                        "ZIP member bytes or decoded contents changed"
                    )
                if _sha256_handle(output_handle) != output_sha256:
                    raise ValueError(
                        "Output changed between write and validation"
                    )
                temporary_identity = _inode_identity(
                    os.fstat(output_handle.fileno())
                )
                path_stat = temporary_path.stat(follow_symlinks=False)
                if (
                    not stat.S_ISREG(path_stat.st_mode)
                    or _inode_identity(path_stat) != temporary_identity
                ):
                    raise OSError(
                        "Temporary output path changed before publication"
                    )
                _publish_new_file(
                    temporary_path,
                    destination_path,
                    expected_identity=temporary_identity,
                )
                published = True
            temporary_cleanup_succeeded = (
                _cleanup_published_temporary(temporary_path)
            )
        finally:
            if not published:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass

    manifest_sha256 = _manifest_sha256(source_manifest)
    stored_crc_mismatches = [
        member["name"]
        for member in source_manifest
        if member["compression"] == zipfile.ZIP_STORED
        and not member["declared_crc32_matches"]
    ]
    return {
        "schema_version": 1,
        "operation": "zip_eocd_comment_cache_bust",
        "source": {
            "name": source_path.name,
            "size_bytes": source_before.st_size,
            "sha256": source_sha256,
            "comment_length_bytes": len(eocd.comment),
            "comment_sha256": hashlib.sha256(eocd.comment).hexdigest(),
        },
        "output": {
            "name": destination_path.name,
            "size_bytes": output_size,
            "sha256": output_sha256,
            "comment_utf8": marker,
            "comment_length_bytes": len(marker_bytes),
            "comment_sha256": hashlib.sha256(marker_bytes).hexdigest(),
        },
        "invariants": {
            "source_opened_read_only": True,
            "artifact_hash_changed": source_sha256 != output_sha256,
            "immutable_prefix_length_bytes": eocd.immutable_prefix_length,
            "immutable_prefix_sha256": immutable_prefix_sha256,
            "member_count": len(source_manifest),
            "member_manifest_sha256": manifest_sha256,
            "compressed_and_decoded_member_hashes_match": True,
            "zip64_eocd_present": eocd.uses_zip64,
            "zip64_structures_validated_when_present": True,
            "stored_declared_crc_mismatch_count": len(
                stored_crc_mismatches
            ),
            "stored_declared_crc_mismatch_members": stored_crc_mismatches,
            "compressed_member_crc_validation_strict": True,
            "temporary_cleanup_succeeded": temporary_cleanup_succeeded,
        },
        "members": source_manifest,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new DLC hash by changing only its ZIP EOCD comment. "
            "The destination must not exist."
        )
    )
    parser.add_argument("source", type=Path, help="source DLC/ZIP")
    parser.add_argument("destination", type=Path, help="new DLC/ZIP")
    parser.add_argument(
        "--marker",
        required=True,
        help="non-empty UTF-8 archive-comment marker (at most 65535 bytes)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        provenance = cache_bust_dlc(
            args.source,
            args.destination,
            args.marker,
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
