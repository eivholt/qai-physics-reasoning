from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_vision_index_aliases import (
    build_alias_index,
    main,
    sha256_file,
    write_json_atomic,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(root: Path) -> tuple[Path, Path, list[str]]:
    target_dir = root / "target"
    inputs = target_dir / "sample_inputs"
    inputs.mkdir(parents=True)
    pixel_hashes: list[str] = []
    pairs: list[dict[str, object]] = []
    for pair_index in range(3):
        payload = bytes([pair_index + 1]) * (16 + pair_index)
        pixel_path = inputs / f"pair_{pair_index:03d}_pixel_values.raw"
        pixel_path.write_bytes(payload)
        pixel_sha = hashlib.sha256(payload).hexdigest()
        pixel_hashes.append(pixel_sha)
        pairs.append(
            {
                "pair_index": pair_index,
                "pixel_values": {
                    "file": (
                        f"sample_inputs/pair_{pair_index:03d}_"
                        "pixel_values.raw"
                    ),
                    "sha256": pixel_sha,
                    "bytes": len(payload),
                },
            }
        )
    target_manifest = target_dir / "video_npu_manifest.json"
    _write_json(target_manifest, {"pairs": pairs, "prompt": "variant"})

    source_index = root / "source_index.json"
    _write_json(
        source_index,
        {
            "schema_version": 1,
            "reference": "bf16_reference",
            "geometry": {"output_shape": [84, 8]},
            "cases": [
                {
                    "case_id": f"source:{pair_index}",
                    "index": pair_index,
                    "manifest_sha256": "a" * 64,
                    "pair_index": pair_index,
                    "pixel_sha256": pixel_sha,
                }
                for pair_index, pixel_sha in enumerate(pixel_hashes)
            ],
        },
    )
    return source_index, target_manifest, pixel_hashes


class VisionIndexAliasTests(unittest.TestCase):
    def test_exact_pixel_aliases_are_provenance_rich_and_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target, pixel_hashes = _fixture(root)
            derived = build_alias_index(source, [("barrier_shuffle", target)])

            self.assertEqual(derived["schema_version"], 2)
            self.assertEqual(
                derived["index_kind"],
                "pixel_verified_vision_batch_aliases",
            )
            self.assertEqual(len(derived["cases"]), 6)
            aliases = derived["cases"][3:]
            target_sha = sha256_file(target)
            self.assertEqual(
                [record["index"] for record in aliases],
                [0, 1, 2],
            )
            self.assertEqual(
                [record["pair_index"] for record in aliases],
                [0, 1, 2],
            )
            self.assertEqual(
                [record["pixel_sha256"] for record in aliases],
                pixel_hashes,
            )
            self.assertTrue(
                all(
                    record["manifest_sha256"] == target_sha
                    for record in aliases
                )
            )
            source_sha = sha256_file(source)
            for record in aliases:
                provenance = record["alias_provenance"]
                self.assertEqual(
                    provenance["match_kind"],
                    "exact_pair_index_and_pixel_sha256",
                )
                self.assertEqual(
                    provenance["source_index_sha256"],
                    source_sha,
                )
                self.assertEqual(
                    provenance["target_manifest_sha256"],
                    target_sha,
                )
            serialized = json.dumps(derived)
            self.assertNotIn(str(root), serialized)
            self.assertFalse(derived["security"]["contains_local_paths"])

    def test_ambiguous_source_match_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target, _ = _fixture(root)
            value = json.loads(source.read_text(encoding="utf-8"))
            duplicate = dict(value["cases"][0])
            duplicate.update(
                {
                    "case_id": "different-source",
                    "index": 3,
                    "manifest_sha256": "b" * 64,
                }
            )
            value["cases"].append(duplicate)
            _write_json(source, value)
            with self.assertRaisesRegex(ValueError, "Ambiguous source"):
                build_alias_index(source, [("variant", target)])

    def test_target_pixel_file_is_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target, _ = _fixture(root)
            pixel_path = target.parent / "sample_inputs/pair_001_pixel_values.raw"
            pixel_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "byte count changed|SHA changed"):
                build_alias_index(source, [("variant", target)])

    def test_check_only_does_not_write_and_output_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target, _ = _fixture(root)
            output = root / "derived.json"
            exit_code = main(
                [
                    str(source),
                    str(output),
                    "--target",
                    f"variant={target}",
                    "--check-only",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertFalse(output.exists())

            derived = build_alias_index(source, [("variant", target)])
            write_json_atomic(output, derived)
            first_sha = sha256_file(output)
            write_json_atomic(output, derived)
            self.assertEqual(sha256_file(output), first_sha)
            changed = dict(derived)
            changed["reference"] = "different"
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                write_json_atomic(output, changed)


if __name__ == "__main__":
    unittest.main()
