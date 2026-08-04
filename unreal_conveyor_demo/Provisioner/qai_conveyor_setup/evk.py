from __future__ import annotations

import concurrent.futures
import ipaddress
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .logging_support import InstallLogger, user_data_dir
from .payload import Payload, sha256


EVK_MODEL_ID = "local/cosmos-reason2-2b"
EVK_MODEL_IMPORT_ID = "local/cosmos-reason2-2b:Q4_0"
EVK_PORT = 18181
SSH_PORT = 22
COMMON_ADDRESSES = (
    "192.168.1.158",
    "192.168.1.92",
    "192.168.42.1",  # common USB Ethernet gadget subnet
    "192.168.7.2",   # common USB Ethernet gadget subnet
    "10.0.0.1",
)
KNOWN_GENIEX_ROOTS = (
    "/home/ubuntu/geniex-cosmos-v0317-native-video-r1",
    "/home/ubuntu/geniex-cosmos-v0317",
)
KNOWN_DATA_ROOTS = (
    "/home/ubuntu/geniex-cosmos-v0317-data",
    "/home/ubuntu/geniex-cosmos-data",
)


def _tcp_open(host: str, port: int, timeout: float = 0.18) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _local_ipv4_addresses() -> set[str]:
    addresses: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(info[4][0])
    except OSError:
        pass
    command = ["ipconfig"] if sys.platform == "win32" else ["ifconfig"]
    try:
        output = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return addresses
    addresses.update(re.findall(r"(?<![\d.])(?:10|172|192)\.\d+\.\d+\.\d+(?![\d.])", output))
    return {address for address in addresses if not address.startswith("127.")}


def discover_ssh_candidates(logger: InstallLogger, explicit: str | None = None) -> list[str]:
    ordered: list[str] = []
    if explicit:
        ordered.append(explicit)
    for name in ("iq9075-evk.local", "qcs9075-evk.local", "iq9075-evk"):
        try:
            ordered.append(socket.gethostbyname(name))
        except OSError:
            pass
    ordered.extend(COMMON_ADDRESSES)

    local_addresses = _local_ipv4_addresses()
    scan: list[str] = []
    for local in local_addresses:
        address = ipaddress.ip_address(local)
        if address.is_private:
            network = ipaddress.ip_network(f"{local}/24", strict=False)
            scan.extend(str(candidate) for candidate in network.hosts() if str(candidate) != local)
    scan = list(dict.fromkeys(scan))[:762]
    logger.event("evk_discovery_scan", local_addresses=sorted(local_addresses), candidate_count=len(scan))
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(_tcp_open, address, SSH_PORT): address for address in scan}
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                ordered.append(futures[future])
    result = list(dict.fromkeys(ordered))
    logger.event("evk_discovery_complete", ssh_candidates=result)
    return result


def direct_model_ready(host: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{EVK_PORT}/v1/models", timeout=3) as response:
            return response.status == 200 and EVK_MODEL_ID in response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError):
        return False


@dataclass
class EvkConnection:
    host: str
    client: object

    def run(self, command: str, timeout: float = 120.0, check: bool = True) -> str:
        stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)  # type: ignore[attr-defined]
        del stdin
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
        if check and code:
            raise RuntimeError(f"EVK command failed ({code}): {command}\n{out}\n{err}")
        return out + err

    def close(self) -> None:
        self.client.close()  # type: ignore[attr-defined]


def connect_evk(
    candidates: list[str],
    username: str,
    password: str,
    logger: InstallLogger,
) -> EvkConnection | None:
    try:
        import paramiko
    except ImportError as error:
        raise RuntimeError("The provisioner build is missing its required Paramiko SSH component") from error

    for host in candidates:
        if not _tcp_open(host, SSH_PORT, timeout=0.5):
            continue
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=host,
                port=SSH_PORT,
                username=username,
                password=password,
                look_for_keys=True,
                allow_agent=True,
                timeout=4,
                auth_timeout=5,
                banner_timeout=5,
            )
            connection = EvkConnection(host, client)
            identity = connection.run(
                "set -e; printf 'arch='; uname -m; printf 'model='; "
                "if [ -r /proc/device-tree/model ]; then tr -d '\\000' </proc/device-tree/model; fi; printf '\\n'",
                timeout=10,
            )
            lower = identity.lower()
            if "aarch64" not in lower or not any(name in lower for name in ("iq", "qcs", "qualcomm", "dragonwing")):
                logger.event("evk_identity_rejected", level="WARNING", host=host, identity=identity)
                connection.close()
                continue
            logger.event("evk_connected", host=host, identity=identity)
            return connection
        except Exception as error:  # Paramiko exposes several transport-specific exception classes.
            logger.event("evk_connect_failed", level="WARNING", host=host, error=error)
            client.close()
    return None


def _find_existing_runtime(connection: EvkConnection) -> tuple[str, str] | None:
    roots = " ".join(shlex.quote(path) for path in KNOWN_GENIEX_ROOTS)
    data_roots = " ".join(shlex.quote(path) for path in KNOWN_DATA_ROOTS)
    output = connection.run(
        f"for p in {roots}; do [ -x \"$p/geniex-grammar\" ] && echo ROOT=$p && break; done; "
        f"for p in {data_roots}; do [ -d \"$p\" ] && echo DATA=$p && break; done",
        timeout=15,
        check=False,
    )
    root_match = re.search(r"^ROOT=(.+)$", output, re.MULTILINE)
    data_match = re.search(r"^DATA=(.+)$", output, re.MULTILINE)
    if root_match and data_match:
        return root_match.group(1).strip(), data_match.group(1).strip()
    return None


def _upload(connection: EvkConnection, source: Path, destination: str, logger: InstallLogger) -> None:
    sftp = connection.client.open_sftp()  # type: ignore[attr-defined]
    last_report = 0

    def progress(copied: int, total: int) -> None:
        nonlocal last_report
        if copied == total or copied - last_report >= 256 * 1024 * 1024:
            logger.event("evk_upload_progress", file=source.name, copied=copied, total=total)
            last_report = copied

    try:
        sftp.put(str(source), destination, callback=progress, confirm=True)
    finally:
        sftp.close()


def _model_import_command(root: str, data: str, models: str, digest: str) -> str:
    """Build the idempotent, local-only model import command for a fresh EVK."""
    importing = data + ".importing"
    marker = data + "/.qai-conveyor-archive-sha256"
    executable = root + "/geniex-grammar"
    return (
        "set -e; "
        f"if [ \"$(cat {shlex.quote(marker)} 2>/dev/null || true)\" != {shlex.quote(digest)} ]; then "
        f"rm -rf {shlex.quote(importing)}; mkdir -p {shlex.quote(importing)}; "
        f"env GENIEX_DATADIR={shlex.quote(importing)} "
        f"LD_LIBRARY_PATH={shlex.quote(root + ':' + root + '/llama_cpp')} "
        f"GENIEX_PLUGIN_PATH={shlex.quote(root)} "
        f"{shlex.quote(executable)} --data-dir {shlex.quote(importing)} "
        f"pull {shlex.quote(EVK_MODEL_IMPORT_ID)} --model-hub localfs "
        f"--local-path {shlex.quote(models)} --model-type vlm; "
        f"test -n \"$(find {shlex.quote(importing)} -mindepth 1 -maxdepth 2 -print -quit)\"; "
        f"printf '%s\\n' {shlex.quote(digest)} >{shlex.quote(importing + '/.qai-conveyor-archive-sha256')}; "
        f"rm -rf {shlex.quote(data)}; mv {shlex.quote(importing)} {shlex.quote(data)}; "
        "fi"
    )


def _deploy_bundled_runtime(connection: EvkConnection, payload: Payload, logger: InstallLogger) -> tuple[str, str]:
    spec = payload.spec("evk-geniex-bundle")
    packaged_archive = payload.packaged_path("evk-geniex-bundle")
    if packaged_archive and payload.validate(packaged_archive, spec):
        local_archive = packaged_archive
        logger.event("evk_bundle_used_in_place", path=local_archive)
    else:
        local_archive = payload.ensure_file(
            "evk-geniex-bundle",
            user_data_dir() / "Downloads" / Path(str(spec.get("path", "evk-geniex-bundle.tar.gz"))).name,
        )
    digest = sha256(local_archive)
    release_root = f"/home/ubuntu/qai-conveyor/releases/{digest[:16]}"
    remote_archive = f"/home/ubuntu/qai-conveyor/cache/{digest}.tar.gz"
    connection.run("mkdir -p /home/ubuntu/qai-conveyor/cache /home/ubuntu/qai-conveyor/releases")
    remote_hash = connection.run(
        f"if [ -f {shlex.quote(remote_archive)} ]; then sha256sum {shlex.quote(remote_archive)} | cut -d' ' -f1; fi",
        check=False,
    ).strip()
    if remote_hash != digest:
        temporary = remote_archive + ".uploading"
        _upload(connection, local_archive, temporary, logger)
        connection.run(
            f"test \"$(sha256sum {shlex.quote(temporary)} | cut -d' ' -f1)\" = {shlex.quote(digest)}; "
            f"mv {shlex.quote(temporary)} {shlex.quote(remote_archive)}"
        )
    layout = connection.run(
        f"set -e; mkdir -p {shlex.quote(release_root)}; "
        f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(release_root)}; "
        f"test -x {shlex.quote(release_root + '/geniex/geniex-grammar')}; "
        f"if [ -d {shlex.quote(release_root + '/data')} ]; then echo DATA_READY; "
        f"elif [ -d {shlex.quote(release_root + '/models')} ]; then echo MODELS_READY; "
        "else echo 'EVK bundle has neither imported data nor local models' >&2; exit 1; fi"
    )
    root = release_root + "/geniex"
    data = release_root + "/data"
    if "MODELS_READY" in layout and "DATA_READY" not in layout:
        logger.event("evk_model_import_started", release_root=release_root, model=EVK_MODEL_IMPORT_ID)
        output = connection.run(
            _model_import_command(root, data, release_root + "/models", digest),
            timeout=900.0,
        )
        logger.event(
            "evk_model_import_complete",
            release_root=release_root,
            output_tail=output[-2000:],
        )
    logger.event("evk_runtime_installed", release_root=release_root)
    return root, data


def _start_lan_service(connection: EvkConnection, root: str, data: str, logger: InstallLogger) -> None:
    service_log = "/home/ubuntu/qai-conveyor/logs/geniex-conveyor.log"
    pid_file = "/home/ubuntu/qai-conveyor/run/geniex-conveyor.pid"
    command = (
        "set -e; mkdir -p /home/ubuntu/qai-conveyor/logs /home/ubuntu/qai-conveyor/run; "
        f"if [ -r {shlex.quote(pid_file)} ]; then "
        f"old_pid=$(cat {shlex.quote(pid_file)}); "
        "case $old_pid in (*[!0-9]*|'') ;; (*) kill \"$old_pid\" >/dev/null 2>&1 || true; esac; "
        "fi; "
        f"nohup env GENIEX_DATADIR={shlex.quote(data)} "
        f"LD_LIBRARY_PATH={shlex.quote(root + ':' + root + '/llama_cpp')} "
        f"GENIEX_PLUGIN_PATH={shlex.quote(root)} "
        f"{shlex.quote(root + '/geniex-grammar')} --skip-update serve "
        f"--host 0.0.0.0:{EVK_PORT} --keepalive 3600 --compute npu --nctx 4096 --ngl -1 "
        f">{shlex.quote(service_log)} 2>&1 </dev/null & "
        f"echo $! >{shlex.quote(pid_file)}"
    )
    connection.run(command)
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        models = connection.run(
            f"curl -sf --max-time 4 http://127.0.0.1:{EVK_PORT}/v1/models || true",
            timeout=10,
            check=False,
        )
        if EVK_MODEL_ID in models:
            logger.event("evk_service_ready_remote", host=connection.host, model=EVK_MODEL_ID)
            return
        time.sleep(1.0)
    tail = connection.run(f"tail -120 {shlex.quote(service_log)}", check=False)
    raise TimeoutError(f"EVK GenieX did not expose {EVK_MODEL_ID} within 120 seconds:\n{tail}")


def ensure_evk(
    payload: Payload,
    logger: InstallLogger,
    explicit_host: str | None = None,
    required: bool = False,
) -> str | None:
    username = payload.credentials.get("evk_username", "ubuntu")
    password = payload.credentials.get("evk_password", "ubuntu")
    logger.add_secrets((password,))
    if explicit_host and direct_model_ready(explicit_host):
        logger.event("evk_service_reused", host=explicit_host, url=f"http://{explicit_host}:{EVK_PORT}")
        return f"http://{explicit_host}:{EVK_PORT}"
    candidates = discover_ssh_candidates(logger, explicit_host)

    for host in candidates:
        if direct_model_ready(host):
            logger.event("evk_service_reused", host=host, url=f"http://{host}:{EVK_PORT}")
            return f"http://{host}:{EVK_PORT}"

    connection = connect_evk(candidates, username, password, logger)
    if connection is None:
        message = "No compatible IQ-9075/QCS9075 EVK was reachable over LAN or USB Ethernet SSH"
        if required:
            raise ConnectionError(message)
        logger.event("evk_not_found", level="WARNING", message=message)
        return None
    try:
        runtime = _find_existing_runtime(connection)
        if runtime is None:
            runtime = _deploy_bundled_runtime(connection, payload, logger)
        _start_lan_service(connection, runtime[0], runtime[1], logger)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if direct_model_ready(connection.host):
                url = f"http://{connection.host}:{EVK_PORT}"
                logger.event("evk_service_ready", url=url)
                return url
            time.sleep(0.5)
        raise ConnectionError(
            f"EVK model is ready through SSH but {connection.host}:{EVK_PORT} is not reachable from the host; check the EVK/network firewall"
        )
    finally:
        connection.close()
