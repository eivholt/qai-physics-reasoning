from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.sanitize_local_manifest import sanitize_manifest


class SanitizeManifestTests(unittest.TestCase):
    def test_replaces_paths_private_ip_and_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "local.json"
            output = root / "publishable.json"
            fake_token = "hf_" + "A" * 24
            source.write_text(
                json.dumps(
                    {
                        "artifact_dir": "/tmp/private/artifact",
                        "windows_path": r"C:\Users\name\artifact\input.npz",
                        "command": (
                            "python /tmp/private/run.py "
                            r"C:\Users\name\input.npz "
                            "ssh ubuntu@192.168.50.42"
                        ),
                        "api_token": "not-a-real-token",
                        "note": f"credential {fake_token}",
                        "choice_token_ids": {"A": 32},
                    }
                ),
                encoding="utf-8",
            )
            result = sanitize_manifest(source, output)
            serialized = output.read_text(encoding="utf-8")
            self.assertNotIn("192.168.50.42", serialized)
            self.assertNotIn("not-a-real-token", serialized)
            self.assertNotIn(fake_token, serialized)
            self.assertNotIn("/tmp/private/run.py", serialized)
            self.assertNotIn(r"C:\Users\name\input.npz", serialized)
            self.assertNotIn(str(root), serialized)
            self.assertEqual(result["api_token"], "<REDACTED>")
            self.assertEqual(result["choice_token_ids"], {"A": 32})
            self.assertEqual(
                result["publication"]["classification"],
                "publishable_sanitized",
            )

    def test_publishable_host_evidence_requires_weight_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "host.json"
            source.write_text(
                json.dumps(
                    {
                        "model": {
                            "weight_shards": [
                                {"name": "model.safetensors", "size_bytes": 1}
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "requires SHA-256"):
                sanitize_manifest(source, root / "publishable.json")


if __name__ == "__main__":
    unittest.main()
