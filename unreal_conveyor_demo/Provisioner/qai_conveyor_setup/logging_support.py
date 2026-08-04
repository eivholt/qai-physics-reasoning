from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Iterator, Sequence


APP_DIR_NAME = "QaiConveyorDemo"
TOKEN_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{12,}\b"),
    re.compile(r"(?i)(api[_-]?key|token|password)(\s*[=:]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(authorization:\s*bearer\s+)([^\s]+)"),
)


def user_data_dir() -> Path:
    override = os.environ.get("QAI_CONVEYOR_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_DIR_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP_DIR_NAME


def log_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / APP_DIR_NAME
    return user_data_dir() / "Logs"


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class InstallLogger:
    """Dual human/JSONL logger with mandatory credential redaction."""

    def __init__(self, operation: str, secrets: Sequence[str] = ()) -> None:
        self.operation = operation
        self.started = dt.datetime.now(dt.timezone.utc)
        self.root = log_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        stem = f"{operation}-{utc_stamp()}-{os.getpid()}"
        self.text_path = self.root / f"{stem}.log"
        self.json_path = self.root / f"{stem}.jsonl"
        self.secrets = {str(value) for value in secrets if value}
        self.diagnostic_files: list[tuple[Path, str]] = []
        self.event("session_start", operation=operation, argv=self.safe_argv(sys.argv))

    def add_secrets(self, values: Sequence[str]) -> None:
        self.secrets.update(str(value) for value in values if value)

    def add_diagnostic_file(self, path: Path, archive_name: str) -> None:
        """Register a non-secret text file for the next support bundle."""
        self.diagnostic_files.append((path, archive_name))

    def redact(self, value: object) -> str:
        text = str(value)
        for secret in sorted(self.secrets, key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")
        text = TOKEN_PATTERNS[0].sub("[REDACTED_HF_TOKEN]", text)
        text = TOKEN_PATTERNS[1].sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
        text = TOKEN_PATTERNS[2].sub(lambda m: f"{m.group(1)}[REDACTED]", text)
        return text

    def safe_argv(self, argv: Sequence[object]) -> list[str]:
        result: list[str] = []
        redact_next = False
        for item in argv:
            text = str(item)
            if redact_next:
                result.append("[REDACTED]")
                redact_next = False
                continue
            result.append(self.redact(text))
            if text.lower() in {"--password", "--token", "--api-key", "--hf-token"}:
                redact_next = True
        return result

    def event(self, name: str, level: str = "INFO", **fields: object) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        safe_fields = {key: self.redact(value) for key, value in fields.items()}
        record = {
            "ts": now.isoformat(timespec="milliseconds"),
            "level": level,
            "event": name,
            **safe_fields,
        }
        human_fields = " ".join(f"{key}={value}" for key, value in safe_fields.items())
        human = f"{record['ts']} [{level}] {name}{(' ' + human_fields) if human_fields else ''}\n"
        with self.text_path.open("a", encoding="utf-8") as stream:
            stream.write(human)
        with self.json_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(human.rstrip(), flush=True)

    @contextlib.contextmanager
    def phase(self, name: str, **fields: object) -> Iterator[None]:
        started = dt.datetime.now(dt.timezone.utc)
        self.event("phase_start", phase=name, **fields)
        try:
            yield
        except BaseException as error:
            elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
            self.event("phase_failed", level="ERROR", phase=name, elapsed_s=f"{elapsed:.3f}", error=error)
            raise
        else:
            elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
            self.event("phase_complete", phase=name, elapsed_s=f"{elapsed:.3f}")

    def command(
        self,
        args: Sequence[str],
        *,
        timeout: float = 120.0,
        check: bool = True,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        safe_args = self.safe_argv(args)
        self.event("command_start", argv=safe_args, cwd=cwd or "")
        started = dt.datetime.now(dt.timezone.utc)
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        output = self.redact(completed.stdout[-20000:])
        elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
        self.event(
            "command_complete",
            argv=safe_args,
            exit_code=completed.returncode,
            elapsed_s=f"{elapsed:.3f}",
            output=output,
        )
        if check and completed.returncode:
            raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(safe_args)}\n{output}")
        return completed

    def record_exception(self, error: BaseException) -> None:
        self.event("unhandled_exception", level="ERROR", error=error, traceback=traceback.format_exc())

    def _copy_redacted_tail(self, source: Path, destination: Path, max_bytes: int = 4 * 1024 * 1024) -> None:
        with source.open("rb") as stream:
            size = source.stat().st_size
            if size > max_bytes:
                stream.seek(-max_bytes, os.SEEK_END)
            text = stream.read().decode("utf-8", errors="replace")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.redact(text), encoding="utf-8")

    def _runtime_log_roots(self) -> list[Path]:
        if sys.platform == "win32":
            local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            return [
                local / "QaiConveyor" / "Saved" / "Logs",
                local / "QaiConveyor" / "Saved" / "Crashes",
            ]
        if sys.platform == "darwin":
            return [Path.home() / "Library" / "Logs" / "QaiConveyor"]
        return []

    def support_bundle(self, reason: str = "manual") -> Path:
        self.event("support_bundle_start", reason=reason)
        stamp = utc_stamp()
        staging = self.root / f"support-{stamp}-{os.getpid()}"
        staging.mkdir(parents=True, exist_ok=True)
        state = {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "reason": self.redact(reason),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": str(Path(sys.executable)),
            "hostname": socket.gethostname(),
            "operation_log": self.text_path.name,
        }
        (staging / "system.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

        log_files = [
            path
            for path in self.root.rglob("*")
            if path.is_file()
            and staging not in path.parents
            and path.suffix.lower() in {".log", ".jsonl", ".txt"}
        ]
        for path in sorted(log_files, key=lambda item: item.stat().st_mtime, reverse=True)[:30]:
            relative = path.relative_to(self.root)
            archive_name = "__".join(relative.parts)
            self._copy_redacted_tail(path, staging / "logs" / archive_name)

        for root in self._runtime_log_roots():
            if not root.is_dir():
                continue
            runtime_logs = [
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".log", ".txt", ".xml", ".json"}
            ]
            for path in sorted(runtime_logs, key=lambda item: item.stat().st_mtime, reverse=True)[:10]:
                self._copy_redacted_tail(path, staging / "runtime-logs" / path.name)

        for path, archive_name in self.diagnostic_files:
            if path.is_file():
                self._copy_redacted_tail(path, staging / "diagnostics" / Path(archive_name).name)
        runtime = user_data_dir() / "runtime.json"
        if runtime.is_file():
            self._copy_redacted_tail(runtime, staging / "runtime.json")

        bundle = self.root / f"QaiConveyor-support-{stamp}.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in staging.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(staging))
        shutil.rmtree(staging)
        self.event("support_bundle_complete", path=bundle)
        return bundle
