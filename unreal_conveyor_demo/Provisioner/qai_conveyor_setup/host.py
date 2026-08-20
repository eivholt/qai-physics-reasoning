from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .logging_support import InstallLogger, user_data_dir
from .payload import Payload, model_search_paths, sha256


HOST_MODEL_NAME = "Cosmos-Reason2-2B-Parcel-Speed-v1-Q8_0.gguf"
PROJECTOR_NAME = "mmproj-Cosmos-Reason2-2B-Parcel-Speed-v1-F16.gguf"
HOST_MODEL_ID = "Cosmos-Reason2-2B-Parcel-Speed-v1"
HOST_PORT = 18084
HOST_CONTEXT_SIZE = 512
HOST_IMAGE_TOKENS = 112


def endpoint_ready(url: str, expected_model: str | None = None, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/v1/models", timeout=timeout) as response:
            if response.status != 200:
                return False
            body = response.read().decode("utf-8", errors="replace")
            return not expected_model or expected_model in body
    except (OSError, urllib.error.URLError):
        return False


def _has_nvidia_driver() -> bool:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False
    try:
        return subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _runtime_artifact_ids(force_cpu: bool = False) -> tuple[str, ...]:
    if sys.platform == "win32":
        if force_cpu or not _has_nvidia_driver():
            return ("host-runtime-windows-cpu",)
        return ("host-runtime-windows-bin", "host-runtime-windows-cudart")
    if sys.platform == "darwin":
        machine = platform.machine().lower()
        if machine in {"arm64", "aarch64"}:
            return ("host-runtime-macos-arm64",)
        if machine in {"x86_64", "amd64"}:
            return ("host-runtime-macos-x64",)
        raise RuntimeError(f"Unsupported Mac architecture: {machine}")
    raise RuntimeError(f"Unsupported host platform: {sys.platform}")


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if target != destination_resolved and destination_resolved not in target.parents:
                    raise ValueError(f"Runtime archive member escapes destination: {member.filename}")
            bundle.extractall(destination)
        return
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                target = (destination / member.name).resolve()
                if target != destination_resolved and destination_resolved not in target.parents:
                    raise ValueError(f"Runtime archive member escapes destination: {member.name}")
            bundle.extractall(destination, filter="data")
        return
    raise ValueError(f"Unsupported runtime archive type: {archive}")


def ensure_host_runtime(payload: Payload, logger: InstallLogger, force_cpu: bool = False) -> Path:
    artifact_ids = _runtime_artifact_ids(force_cpu=force_cpu)
    archives = []
    for artifact_id in artifact_ids:
        spec = payload.spec(artifact_id)
        cache = user_data_dir() / "Downloads" / Path(str(spec.get("path", artifact_id))).name
        archives.append(payload.ensure_file(artifact_id, cache))
    if sys.platform == "win32":
        variant = "cpu" if artifact_ids == ("host-runtime-windows-cpu",) else "cuda"
        runtime = user_data_dir() / "Runtime" / "Windows" / variant
    else:
        runtime = user_data_dir() / "Runtime" / "Mac"
    marker = runtime / ".archive.sha256"
    archive_hash = ":".join(str(payload.spec(artifact_id)["sha256"]).lower() for artifact_id in artifact_ids)
    executable_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    executable_candidates = list(runtime.rglob(executable_name)) if runtime.is_dir() else []
    if marker.is_file() and marker.read_text(encoding="ascii").strip() == archive_hash and len(executable_candidates) == 1:
        logger.event("host_runtime_reused", executable=executable_candidates[0])
        return executable_candidates[0]

    if runtime.exists():
        expected_root = (user_data_dir() / "Runtime").resolve()
        if expected_root not in runtime.resolve().parents:
            raise RuntimeError(f"Refusing to replace runtime outside application data: {runtime}")
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True)
    for archive in archives:
        _safe_extract(archive, runtime)
    marker.write_text(archive_hash + "\n", encoding="ascii")
    executable_candidates = list(runtime.rglob(executable_name))
    if len(executable_candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {executable_name} in {archives}; found {len(executable_candidates)}"
        )
    executable = executable_candidates[0]
    if sys.platform != "win32":
        executable.chmod(executable.stat().st_mode | 0o111)
    logger.event("host_runtime_installed", executable=executable)
    return executable


def is_cpu_runtime(executable: Path) -> bool:
    return sys.platform == "win32" and "cpu" in {part.lower() for part in executable.parts}


def ensure_host_models(payload: Payload, logger: InstallLogger) -> tuple[Path, Path]:
    model_dir = user_data_dir() / "Models"
    model = payload.ensure_file(
        "host-parcel-speed-v1-q8",
        model_dir / HOST_MODEL_NAME,
        model_search_paths(HOST_MODEL_NAME),
    )
    projector = payload.ensure_file(
        "host-parcel-speed-v1-projector-f16",
        model_dir / PROJECTOR_NAME,
        model_search_paths(PROJECTOR_NAME),
    )
    return model, projector


def start_host_server(
    executable: Path,
    model: Path,
    projector: Path,
    logger: InstallLogger,
) -> str:
    url = f"http://127.0.0.1:{HOST_PORT}"
    if endpoint_ready(url, HOST_MODEL_ID):
        logger.event("host_server_reused", url=url)
        return url

    server_log_dir = logger.root / "Servers"
    server_log_dir.mkdir(parents=True, exist_ok=True)
    server_log = server_log_dir / f"host-server-{int(time.time())}.log"
    args = [
        str(executable),
        "-m",
        str(model),
        "--mmproj",
        str(projector),
        "-ngl",
        "0" if is_cpu_runtime(executable) else "all",
        "-c",
        str(HOST_CONTEXT_SIZE),
        "--host",
        "127.0.0.1",
        "--port",
        str(HOST_PORT),
        "--alias",
        HOST_MODEL_ID,
        "--image-min-tokens",
        str(HOST_IMAGE_TOKENS),
        "--image-max-tokens",
        str(HOST_IMAGE_TOKENS),
        "--flash-attn",
        "on",
        "--no-cache-prompt",
        "--slot-prompt-similarity",
        "0",
    ]
    logger.event("host_server_start", argv=args, log=server_log)
    stream = server_log.open("ab", buffering=0)
    kwargs: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "stdout": stream,
        "stderr": subprocess.STDOUT,
        "cwd": executable.parent,
        "close_fds": True,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(args, **kwargs)  # type: ignore[arg-type]
    finally:
        stream.close()
    (user_data_dir() / "host-server.pid").write_text(str(process.pid) + "\n", encoding="ascii")
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = server_log.read_text(encoding="utf-8", errors="replace")[-12000:]
            raise RuntimeError(f"Host Reason2 server exited with {process.returncode}:\n{tail}")
        if endpoint_ready(url, HOST_MODEL_ID, timeout=3.0):
            logger.event("host_server_ready", url=url, pid=process.pid)
            return url
        time.sleep(1.0)
    tail = server_log.read_text(encoding="utf-8", errors="replace")[-12000:]
    raise TimeoutError(f"Host Reason2 server was not ready within 120 seconds:\n{tail}")


def write_runtime_config(host_url: str, evk_url: str | None, logger: InstallLogger) -> Path:
    configured_evk_url = evk_url or "http://127.0.0.1:18183"
    config = {
        "schema_version": 1,
        "host_server_url": host_url,
        "host_model": HOST_MODEL_ID,
        "evk_server_url": configured_evk_url,
        "evk_model": "local/cosmos-reason2-2b",
        "backend": "host",
        "auto_inference": True,
    }
    destination = user_data_dir() / "runtime.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    logger.event("runtime_config_written", path=destination, host_url=host_url, evk_url=evk_url or "unavailable")
    return destination
