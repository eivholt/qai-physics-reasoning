from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from qai_conveyor_setup import cli, evk, host
from qai_conveyor_setup.host import write_runtime_config
from qai_conveyor_setup.logging_support import InstallLogger
from qai_conveyor_setup.payload import Payload
from qai_conveyor_setup.payload import default_payload_candidates
from qai_conveyor_setup.payload import model_search_paths


class ProvisionerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.environment = mock.patch.dict(os.environ, {"QAI_CONVEYOR_DATA_DIR": str(self.root / "data")})
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary.cleanup()

    def test_logs_and_support_bundle_redact_tokens_and_passwords(self) -> None:
        secret = "hf_abcdefghijklmnopqrstuvwxyz123456"
        logger = InstallLogger("test", secrets=(secret, "personal-password"))
        logger.event(
            "credential_test",
            text=f"token={secret} password=personal-password Authorization: Bearer {secret}",
        )
        server_log = logger.root / "Servers" / "host-server.log"
        server_log.parent.mkdir(parents=True)
        server_log.write_text(f"nested server token={secret}", encoding="utf-8")
        manifest = self.root / "payload_manifest.json"
        manifest.write_text(f'{{"token":"{secret}","schema_version":1}}', encoding="utf-8")
        logger.add_diagnostic_file(manifest, "payload_manifest.json")
        bundle = logger.support_bundle(f"failed with {secret}")
        self.assertTrue(bundle.is_file())
        all_logs = "".join(path.read_text(encoding="utf-8") for path in logger.root.glob("*.log"))
        self.assertNotIn(secret, all_logs)
        self.assertNotIn("personal-password", all_logs)
        with zipfile.ZipFile(bundle) as archive:
            self.assertTrue(any("host-server.log" in name for name in archive.namelist()))
            self.assertIn("diagnostics/payload_manifest.json", archive.namelist())
            contents = "".join(
                archive.read(name).decode("utf-8", errors="replace") for name in archive.namelist()
            )
        self.assertNotIn(secret, contents)
        self.assertNotIn("personal-password", contents)

    def test_payload_requires_exact_sha256(self) -> None:
        payload_root = self.root / "payload"
        artifact = payload_root / "Models" / "tiny.gguf"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"verified model")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        manifest = {
            "artifacts": [
                {
                    "id": "tiny",
                    "path": "Models/tiny.gguf",
                    "bytes": artifact.stat().st_size,
                    "sha256": digest,
                }
            ]
        }
        (payload_root / "payload_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        logger = InstallLogger("test")
        payload = Payload(logger, root=payload_root)
        destination = self.root / "installed" / "tiny.gguf"
        self.assertEqual(payload.ensure_file("tiny", destination), destination)
        self.assertEqual(destination.read_bytes(), b"verified model")
        destination.write_bytes(b"corrupt")
        payload.ensure_file("tiny", destination)
        self.assertEqual(destination.read_bytes(), b"verified model")

    def test_payload_reuses_hash_cache_for_unchanged_file(self) -> None:
        payload_root = self.root / "payload"
        artifact = payload_root / "Models" / "cached.gguf"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"hash once")
        manifest = {
            "artifacts": [
                {
                    "id": "cached",
                    "path": "Models/cached.gguf",
                    "bytes": artifact.stat().st_size,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ]
        }
        (payload_root / "payload_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        destination = self.root / "installed" / "cached.gguf"
        Payload(InstallLogger("test"), root=payload_root).ensure_file("cached", destination)
        reloaded = Payload(InstallLogger("test"), root=payload_root)
        with mock.patch("qai_conveyor_setup.payload.sha256", side_effect=AssertionError("unexpected rehash")):
            self.assertTrue(reloaded.validate(destination, reloaded.spec("cached")))

    def test_curl_transport_resumes_partial_download(self) -> None:
        payload_root = self.root / "payload"
        payload_root.mkdir()
        (payload_root / "payload_manifest.json").write_text('{"artifacts":[]}', encoding="utf-8")
        logger = InstallLogger("test")
        payload = Payload(logger, root=payload_root)
        partial = self.root / "artifact.gguf.download"
        partial.write_bytes(b"partial")
        completed = mock.Mock(returncode=0)
        with (
            mock.patch("qai_conveyor_setup.payload.shutil.which", return_value="curl.exe"),
            mock.patch.object(logger, "command", return_value=completed) as command,
        ):
            self.assertTrue(
                payload._download_with_curl(
                    "https://example.invalid/artifact.gguf",
                    partial,
                    {"User-Agent": "QaiConveyorDemo/0.1"},
                    {"id": "artifact"},
                )
            )
        args = command.call_args.args[0]
        self.assertIn("--continue-at", args)
        self.assertIn("--retry-all-errors", args)
        self.assertEqual(args[-2], str(partial))

    def test_payload_rejects_path_escape(self) -> None:
        payload_root = self.root / "payload"
        payload_root.mkdir()
        manifest = {
            "artifacts": [
                {"id": "escape", "path": "../outside", "bytes": 1, "sha256": "00" * 32}
            ]
        }
        (payload_root / "payload_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        payload = Payload(InstallLogger("test"), root=payload_root)
        with self.assertRaises(ValueError):
            payload.packaged_path("escape")

    def test_runtime_config_is_atomic_and_defaults_to_host(self) -> None:
        logger = InstallLogger("test")
        path = write_runtime_config("http://127.0.0.1:18080", "http://192.168.1.158:18181", logger)
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["backend"], "host")
        self.assertEqual(config["evk_server_url"], "http://192.168.1.158:18181")
        self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_explicit_evk_is_prioritized_and_deduplicated(self) -> None:
        logger = InstallLogger("test")
        with (
            mock.patch.object(evk, "_local_ipv4_addresses", return_value=set()),
            mock.patch.object(evk.socket, "gethostbyname", side_effect=OSError),
        ):
            candidates = evk.discover_ssh_candidates(logger, "192.168.1.158")
        self.assertEqual(candidates[0], "192.168.1.158")
        self.assertEqual(candidates.count("192.168.1.158"), 1)

    def test_empty_model_directory_environment_never_searches_working_directory(self) -> None:
        marker = self.root / "accidental.gguf"
        marker.write_bytes(b"do not use")
        with (
            mock.patch.dict(os.environ, {"QAI_CONVEYOR_MODEL_DIR": ""}),
            mock.patch.object(Path, "cwd", return_value=self.root),
        ):
            self.assertNotIn(marker, model_search_paths(marker.name))

    def test_evk_service_uses_scoped_pid_file_not_process_name_kill(self) -> None:
        connection = mock.Mock()
        connection.host = "192.168.1.158"
        connection.run.side_effect = ["", '{"data":[{"id":"local/cosmos-reason2-2b"}]}']
        evk._start_lan_service(connection, "/opt/geniex", "/opt/data", InstallLogger("test"))
        launch_command = connection.run.call_args_list[0].args[0]
        self.assertIn("geniex-conveyor.pid", launch_command)
        self.assertNotIn("pkill", launch_command)

    def test_evk_model_import_is_local_idempotent_and_atomic(self) -> None:
        command = evk._model_import_command(
            "/opt/qai release/geniex",
            "/opt/qai release/data",
            "/opt/qai release/models",
            "abc123",
        )
        self.assertIn("pull local/cosmos-reason2-2b:Q4_0", command)
        self.assertIn("--model-hub localfs", command)
        self.assertIn("--model-type vlm", command)
        self.assertIn("data.importing", command)
        self.assertIn(".qai-conveyor-archive-sha256", command)
        self.assertNotIn("http://", command)
        self.assertNotIn("https://", command)

    def test_ready_explicit_evk_skips_subnet_discovery(self) -> None:
        payload = mock.Mock()
        payload.credentials = {}
        with (
            mock.patch.object(evk, "direct_model_ready", return_value=True),
            mock.patch.object(evk, "discover_ssh_candidates", side_effect=AssertionError("unexpected scan")),
        ):
            url = evk.ensure_evk(payload, InstallLogger("test"), explicit_host="192.168.1.158")
        self.assertEqual(url, "http://192.168.1.158:18181")

    def test_windows_runtime_uses_cpu_without_nvidia_and_when_forced(self) -> None:
        with (
            mock.patch.object(host.sys, "platform", "win32"),
            mock.patch.object(host, "_has_nvidia_driver", return_value=False),
        ):
            self.assertEqual(host._runtime_artifact_ids(), ("host-runtime-windows-cpu",))
        with (
            mock.patch.object(host.sys, "platform", "win32"),
            mock.patch.object(host, "_has_nvidia_driver", return_value=True),
        ):
            self.assertEqual(
                host._runtime_artifact_ids(),
                ("host-runtime-windows-bin", "host-runtime-windows-cudart"),
            )
            self.assertEqual(host._runtime_artifact_ids(force_cpu=True), ("host-runtime-windows-cpu",))

    def test_renamed_macos_app_bundle_finds_game_executable(self) -> None:
        app = self.root / "Reason2 Conveyor Safety.app"
        executable = app / "Contents" / "MacOS" / "QaiConveyor"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"binary")
        with mock.patch.object(cli.sys, "platform", "darwin"):
            self.assertEqual(cli._find_game(app), executable.resolve())

    def test_windows_installed_layout_finds_payload_next_to_provisioner(self) -> None:
        executable_dir = self.root / "app" / "Provisioner"
        payload_root = self.root / "app" / "Payload"
        executable_dir.mkdir(parents=True)
        payload_root.mkdir(parents=True)
        with mock.patch.object(cli.sys, "executable", str(executable_dir / "qai-conveyor-setup.exe")):
            self.assertIn(payload_root, default_payload_candidates())


if __name__ == "__main__":
    unittest.main()
