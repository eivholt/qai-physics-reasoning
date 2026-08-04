from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

from .logging_support import InstallLogger, user_data_dir


BUFFER_SIZE = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def default_payload_candidates(app_dir: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if app_dir:
        candidates.extend((app_dir / "Payload", app_dir / "Resources" / "Payload"))
    if getattr(sys, "_MEIPASS", None):
        candidates.append(Path(sys._MEIPASS) / "Payload")  # type: ignore[attr-defined]
    executable_dir = Path(sys.executable).resolve().parent
    candidates.extend(
        (
            executable_dir / "Payload",
            executable_dir.parent / "Payload",
            executable_dir.parent / "Resources" / "Payload",
        )
    )
    candidates.append(Path(__file__).resolve().parents[2] / "Payload")
    return candidates


class Payload:
    def __init__(self, logger: InstallLogger, root: Path | None = None, app_dir: Path | None = None) -> None:
        roots = [root] if root else default_payload_candidates(app_dir)
        self.root = next((candidate.resolve() for candidate in roots if candidate and candidate.is_dir()), None)
        if self.root is None:
            raise FileNotFoundError(f"Installer payload not found; checked: {roots}")
        manifest_path = self.root / "payload_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Release payload manifest is missing: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.artifacts = {entry["id"]: entry for entry in self.manifest.get("artifacts", [])}
        self.logger = logger
        self.verification_cache_path = user_data_dir() / "State" / "artifact-verification-v1.json"
        self.verification_cache = self._load_verification_cache()
        self.credentials = self._load_credentials()
        logger.add_secrets([str(value) for value in self.credentials.values()])
        logger.add_diagnostic_file(manifest_path, "payload_manifest.json")
        logger.event("payload_loaded", root=self.root, artifact_count=len(self.artifacts))

    def _load_verification_cache(self) -> dict[str, dict[str, object]]:
        try:
            values = json.loads(self.verification_cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        entries = values.get("entries", {}) if isinstance(values, dict) else {}
        return entries if isinstance(entries, dict) else {}

    @staticmethod
    def _cache_key(path: Path) -> str:
        return os.path.normcase(str(path.resolve()))

    def _remember_verified(self, path: Path, spec: dict[str, object]) -> None:
        stat = path.stat()
        self.verification_cache[self._cache_key(path)] = {
            "artifact": str(spec.get("id", "")),
            "sha256": str(spec.get("sha256", "")).lower(),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        self.verification_cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.verification_cache_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"schema_version": 1, "entries": self.verification_cache}, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.verification_cache_path)

    def _load_credentials(self) -> dict[str, str]:
        path = self.root / "credentials.json"
        if not path.is_file():
            return {}
        values = json.loads(path.read_text(encoding="utf-8"))
        return {str(key): str(value) for key, value in values.items() if value is not None}

    def spec(self, artifact_id: str) -> dict[str, object]:
        try:
            return self.artifacts[artifact_id]
        except KeyError as error:
            raise KeyError(f"Release payload has no artifact named {artifact_id}") from error

    def validate(self, path: Path, spec: dict[str, object], *, allow_cache: bool = True) -> bool:
        if not path.is_file():
            return False
        stat = path.stat()
        expected_bytes = int(spec.get("bytes", 0) or 0)
        if expected_bytes and stat.st_size != expected_bytes:
            self.logger.event("artifact_size_mismatch", level="WARNING", path=path, expected=expected_bytes, actual=stat.st_size)
            return False
        expected_hash = str(spec.get("sha256", ""))
        if not expected_hash:
            raise ValueError(f"Artifact {spec.get('id')} has no SHA-256 in the release manifest")
        cached = self.verification_cache.get(self._cache_key(path), {})
        if allow_cache and (
            cached.get("artifact") == str(spec.get("id", ""))
            and cached.get("sha256") == expected_hash.lower()
            and cached.get("bytes") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
        ):
            self.logger.event("artifact_verification_cache_hit", artifact=spec.get("id"), path=path, bytes=stat.st_size)
            return True
        actual_hash = sha256(path)
        if actual_hash.lower() != expected_hash.lower():
            self.logger.event("artifact_hash_mismatch", level="WARNING", path=path, expected=expected_hash, actual=actual_hash)
            return False
        self._remember_verified(path, spec)
        self.logger.event("artifact_verified", artifact=spec.get("id"), path=path, bytes=stat.st_size)
        return True

    def packaged_path(self, artifact_id: str) -> Path | None:
        spec = self.spec(artifact_id)
        relative = str(spec.get("path", ""))
        if not relative:
            return None
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise ValueError(f"Artifact path escapes payload: {relative}")
        return path

    def ensure_file(
        self,
        artifact_id: str,
        destination: Path,
        search_paths: Iterable[Path] = (),
    ) -> Path:
        spec = self.spec(artifact_id)
        if self.validate(destination, spec):
            return destination
        candidates = [self.packaged_path(artifact_id), *search_paths]
        for candidate in candidates:
            if candidate and self.validate(candidate, spec):
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".copying")
                shutil.copy2(candidate, temporary)
                if not self.validate(temporary, spec, allow_cache=False):
                    temporary.unlink(missing_ok=True)
                    raise IOError(f"Copied artifact failed verification: {artifact_id}")
                os.replace(temporary, destination)
                self.verification_cache.pop(self._cache_key(temporary), None)
                self._remember_verified(destination, spec)
                return destination

        url = str(spec.get("download_url", ""))
        if not url:
            raise FileNotFoundError(
                f"Required artifact {artifact_id} was not found locally or in the installer payload, and no download URL was released"
            )
        self._download(url, destination, spec)
        return destination

    def _download(self, url: str, destination: Path, spec: dict[str, object]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".download")
        headers = {"User-Agent": "QaiConveyorDemo/0.1"}
        token = self.credentials.get("huggingface_token", "")
        if token and "huggingface.co" in url:
            headers["Authorization"] = f"Bearer {token}"
        expected_bytes = int(spec.get("bytes", 0) or 0)
        resume_at = temporary.stat().st_size if temporary.is_file() else 0
        remaining_bytes = max(expected_bytes - resume_at, 0) if expected_bytes else 0
        free_bytes = shutil.disk_usage(destination.parent).free
        reserve_bytes = 512 * 1024 * 1024
        if remaining_bytes and free_bytes < remaining_bytes + reserve_bytes:
            raise OSError(
                f"Not enough free space to download {spec.get('id')}: "
                f"need {remaining_bytes + reserve_bytes} bytes including reserve, have {free_bytes}"
            )
        self.logger.event(
            "artifact_disk_preflight",
            artifact=spec.get("id"),
            free_bytes=free_bytes,
            remaining_bytes=remaining_bytes,
        )
        if resume_at and expected_bytes and resume_at == expected_bytes:
            if self.validate(temporary, spec, allow_cache=False):
                os.replace(temporary, destination)
                self.verification_cache.pop(self._cache_key(temporary), None)
                self._remember_verified(destination, spec)
                self.logger.event("artifact_download_complete", artifact=spec.get("id"), destination=destination)
                return
            temporary.unlink()
            resume_at = 0
        elif expected_bytes and resume_at > expected_bytes:
            temporary.unlink()
            resume_at = 0
        self.logger.event(
            "artifact_download_start",
            artifact=spec.get("id"),
            url=url,
            destination=destination,
            resume_bytes=resume_at,
        )
        downloaded_with_curl = self._download_with_curl(url, temporary, headers, spec)
        if not downloaded_with_curl:
            resume_at = temporary.stat().st_size if temporary.is_file() else 0
            if resume_at:
                headers["Range"] = f"bytes={resume_at}-"
            request = urllib.request.Request(url, headers=headers)
            self.logger.event(
                "artifact_download_transport",
                artifact=spec.get("id"),
                transport="python-urllib",
                resume_bytes=resume_at,
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    partial_response = getattr(response, "status", 200) == 206
                    mode = "ab" if resume_at and partial_response else "wb"
                    copied = resume_at if mode == "ab" else 0
                    if resume_at and mode == "wb":
                        self.logger.event(
                            "artifact_download_restart",
                            level="WARNING",
                            artifact=spec.get("id"),
                            reason="server ignored byte range",
                        )
                    with temporary.open(mode) as output:
                        next_report = ((copied // (512 * 1024 * 1024)) + 1) * (512 * 1024 * 1024)
                        while True:
                            block = response.read(BUFFER_SIZE)
                            if not block:
                                break
                            output.write(block)
                            copied += len(block)
                            if copied >= next_report:
                                self.logger.event("artifact_download_progress", artifact=spec.get("id"), bytes=copied)
                                next_report += 512 * 1024 * 1024
            except urllib.error.HTTPError as error:
                raise RuntimeError(
                    f"Download failed for {spec.get('id')}: HTTP {error.code}; "
                    f"the partial file was retained for a resumable retry"
                ) from error
        if not self.validate(temporary, spec, allow_cache=False):
            temporary.unlink(missing_ok=True)
            raise IOError(f"Downloaded artifact failed verification: {spec.get('id')}")
        os.replace(temporary, destination)
        self.verification_cache.pop(self._cache_key(temporary), None)
        self._remember_verified(destination, spec)
        self.logger.event("artifact_download_complete", artifact=spec.get("id"), destination=destination)

    def _download_with_curl(
        self,
        url: str,
        temporary: Path,
        headers: dict[str, str],
        spec: dict[str, object],
    ) -> bool:
        curl = shutil.which("curl")
        if not curl:
            return False
        args = [
            curl,
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "5",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "30",
            "--speed-limit",
            "1024",
            "--speed-time",
            "120",
        ]
        if temporary.is_file() and temporary.stat().st_size:
            args.extend(("--continue-at", "-"))
        for name, value in headers.items():
            if name.lower() == "user-agent":
                args.extend(("--user-agent", value))
            else:
                args.extend(("--header", f"{name}: {value}"))
        args.extend(("--output", str(temporary), url))
        self.logger.event(
            "artifact_download_transport",
            artifact=spec.get("id"),
            transport="curl",
            executable=curl,
            resume_bytes=temporary.stat().st_size if temporary.is_file() else 0,
        )
        completed = self.logger.command(args, timeout=6 * 60 * 60, check=False)
        if completed.returncode == 0:
            return True
        self.logger.event(
            "artifact_curl_failed",
            level="WARNING",
            artifact=spec.get("id"),
            exit_code=completed.returncode,
            message="Falling back to the built-in resumable downloader",
        )
        return False


def model_search_paths(filename: str) -> list[Path]:
    roots: list[Path] = []
    configured_root = os.environ.get("QAI_CONVEYOR_MODEL_DIR")
    if configured_root:
        roots.append(Path(configured_root).expanduser())
    roots.extend(
        [
        user_data_dir() / "Models",
        Path.home() / "models",
        Path.home() / ".cache" / "huggingface" / "hub",
        ]
    )
    results: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        direct = root / filename
        if direct.is_file():
            results.append(direct)
        if root.name == "hub":
            results.extend(root.glob(f"models--*--*\u002fsnapshots\u002f*\u002f{filename}"))
    return results
