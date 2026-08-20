from __future__ import annotations

import base64
import concurrent.futures
import ipaddress
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
EVK_PORT = 18183
EVK_SPEED_ARTIFACT_ID = "evk-parcel-speed-v1-geniex"
EVK_RUNTIME_ARTIFACT_ID = "evk-geniex-runtime"
EVK_SPEED_RELEASE_ID = "parcel-speed-v1-448x256-20260819"
EVK_SPEED_BUNDLE_NAME = (
    "cosmos-reason2-parcel-speed-v1-cl512-256x448-w8text-geniex-qairt245-os19-r1"
)
EVK_SPEED_DATA_DIR = "/home/ubuntu/qai-conveyor/geniex-data/cosmos-reason2-parcel-speed-v1"
EVK_RUNTIME_DIR = "/home/ubuntu/qai-conveyor/runtime/geniex-v0317-qairt245"
SSH_PORT = 22
COMMON_ADDRESSES = (
    "192.168.1.158",
    "192.168.1.92",
    "192.168.42.1",
    "192.168.7.2",
    "10.0.0.1",
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
        output = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return addresses
    addresses.update(
        re.findall(r"(?<![\d.])(?:10|172|192)\.\d+\.\d+\.\d+(?![\d.])", output)
    )
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
            scan.extend(
                str(candidate) for candidate in network.hosts() if str(candidate) != local
            )
    scan = list(dict.fromkeys(scan))[:762]
    logger.event(
        "evk_discovery_scan",
        local_addresses=sorted(local_addresses),
        candidate_count=len(scan),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = {
            executor.submit(_tcp_open, address, SSH_PORT): address for address in scan
        }
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                ordered.append(futures[future])
    result = list(dict.fromkeys(ordered))
    logger.event("evk_discovery_complete", ssh_candidates=result)
    return result


def direct_model_ready(host: str, port: int = EVK_PORT) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=3) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status == 200 and EVK_MODEL_ID in body
    except (OSError, urllib.error.URLError):
        return False


def available_evk_service_url(host: str) -> str | None:
    """Return the promoted direct-image GenieX endpoint when it is ready."""
    if direct_model_ready(host):
        return f"http://{host}:{EVK_PORT}"
    return None


@dataclass
class EvkConnection:
    host: str
    client: object

    def run(self, command: str, timeout: float = 120.0, check: bool = True) -> str:
        stdin, stdout, stderr = self.client.exec_command(  # type: ignore[attr-defined]
            command, timeout=timeout
        )
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
        raise RuntimeError(
            "The provisioner build is missing its required Paramiko SSH component"
        ) from error

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
            if "aarch64" not in lower or not any(
                name in lower for name in ("iq", "qcs", "qualcomm", "dragonwing")
            ):
                logger.event(
                    "evk_identity_rejected",
                    level="WARNING",
                    host=host,
                    identity=identity,
                )
                connection.close()
                continue
            logger.event("evk_connected", host=host, identity=identity)
            return connection
        except Exception as error:
            logger.event("evk_connect_failed", level="WARNING", host=host, error=error)
            client.close()
    return None


def _upload(
    connection: EvkConnection,
    source: Path,
    destination: str,
    logger: InstallLogger,
) -> None:
    sftp = connection.client.open_sftp()  # type: ignore[attr-defined]
    last_report = 0

    def progress(copied: int, total: int) -> None:
        nonlocal last_report
        if copied == total or copied - last_report >= 256 * 1024 * 1024:
            logger.event(
                "evk_upload_progress", file=source.name, copied=copied, total=total
            )
            last_report = copied

    try:
        sftp.put(str(source), destination, callback=progress, confirm=True)
    finally:
        sftp.close()


def _validated_payload_file(
    payload: Payload,
    artifact_id: str,
    logger: InstallLogger,
) -> Path:
    """Use an embedded artifact in place so large payloads are not duplicated."""
    spec = payload.spec(artifact_id)
    packaged = payload.packaged_path(artifact_id)
    if packaged and payload.validate(packaged, spec):
        return packaged
    cache = user_data_dir() / "Downloads" / Path(str(spec["path"])).name
    logger.event("evk_release_artifact_cache", artifact=artifact_id, path=cache)
    return payload.ensure_file(artifact_id, cache)


def _geniex_systemd_unit(runtime_root: str, data_dir: str) -> str:
    qairt = "/opt/qairt/2.45.0.260326"
    target = "aarch64-oe-linux-gcc11.2"
    return f"""[Unit]
Description=QAI Conveyor speed-v1 Reason2 GenieX QAIRT worker
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=3

[Service]
Type=simple
User=ubuntu
Group=ubuntu
UMask=0022
WorkingDirectory={data_dir}
ExecStartPre=/usr/bin/test -x {runtime_root}/geniex-grammar
ExecStartPre=/usr/bin/test -d {data_dir}
ExecStart={runtime_root}/geniex-grammar --skip-update serve --host 0.0.0.0:{EVK_PORT} --keepalive 3600 --compute npu --nctx 512 --ngl -1
Restart=always
RestartSec=5
TimeoutStartSec=120
TimeoutStopSec=30
KillMode=control-group
Environment=GENIEX_DATADIR={data_dir}
Environment=GENIEX_PLUGIN_PATH={runtime_root}
Environment=PATH={qairt}/bin/{target}:{qairt}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=LD_LIBRARY_PATH={runtime_root}:{runtime_root}/qairt:{runtime_root}/llama_cpp:{qairt}/lib/{target}:/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu
Environment=ADSP_LIBRARY_PATH={qairt}/lib/hexagon-v73/unsigned
StandardOutput=journal
StandardError=journal
SyslogIdentifier=qai-conveyor-geniex

[Install]
WantedBy=multi-user.target
"""


def _model_import_command(root: str, data: str, models: str, digest: str) -> str:
    """Build the idempotent, local-only speed-v1 model import command."""
    importing = data + ".importing"
    marker = data + "/.qai-conveyor-archive-sha256"
    executable = root + "/geniex-grammar"
    return (
        "set -e; "
        f"if [ \"$(cat {shlex.quote(marker)} 2>/dev/null || true)\" != {shlex.quote(digest)} ]; then "
        f"rm -rf {shlex.quote(importing)}; mkdir -p {shlex.quote(importing)}; "
        f"env GENIEX_DATADIR={shlex.quote(importing)} "
        f"LD_LIBRARY_PATH={shlex.quote(root + ':' + root + '/qairt:' + root + '/llama_cpp')} "
        f"GENIEX_PLUGIN_PATH={shlex.quote(root)} "
        f"{shlex.quote(executable)} --data-dir {shlex.quote(importing)} "
        f"pull {shlex.quote(EVK_MODEL_IMPORT_ID)} --model-hub localfs "
        f"--local-path {shlex.quote(models)} --model-type vlm; "
        f"test -n \"$(find {shlex.quote(importing)} -mindepth 1 -maxdepth 2 -print -quit)\"; "
        f"printf '%s\\n' {shlex.quote(digest)} >{shlex.quote(importing + '/.qai-conveyor-archive-sha256')}; "
        f"rm -rf {shlex.quote(data)}; mv {shlex.quote(importing)} {shlex.quote(data)}; "
        "fi"
    )


def _deploy_runtime(
    connection: EvkConnection,
    payload: Payload,
    logger: InstallLogger,
) -> str:
    archive = _validated_payload_file(payload, EVK_RUNTIME_ARTIFACT_ID, logger)
    digest = sha256(archive)
    remote_archive = f"/home/ubuntu/qai-conveyor/cache/{digest}.tar.gz"
    marker = f"{EVK_RUNTIME_DIR}/.qai-conveyor-runtime-sha256"
    connection.run("mkdir -p /home/ubuntu/qai-conveyor/cache /home/ubuntu/qai-conveyor/runtime")
    remote_hash = connection.run(
        f"if [ -f {shlex.quote(remote_archive)} ]; then sha256sum {shlex.quote(remote_archive)} | cut -d' ' -f1; fi",
        check=False,
    ).strip()
    if remote_hash != digest:
        temporary = remote_archive + ".uploading"
        _upload(connection, archive, temporary, logger)
        connection.run(
            f"test \"$(sha256sum {shlex.quote(temporary)} | cut -d' ' -f1)\" = {shlex.quote(digest)}; "
            f"mv {shlex.quote(temporary)} {shlex.quote(remote_archive)}"
        )
    importing = EVK_RUNTIME_DIR + ".importing"
    connection.run(
        "set -e; "
        f"if [ \"$(cat {shlex.quote(marker)} 2>/dev/null || true)\" != {shlex.quote(digest)} ]; then "
        f"rm -rf -- {shlex.quote(importing)}; mkdir -p {shlex.quote(importing)}; "
        f"tar -xzf {shlex.quote(remote_archive)} --strip-components=1 -C {shlex.quote(importing)}; "
        f"test -x {shlex.quote(importing + '/geniex-grammar')}; "
        f"printf '%s\\n' {shlex.quote(digest)} >{shlex.quote(importing + '/.qai-conveyor-runtime-sha256')}; "
        f"rm -rf -- {shlex.quote(EVK_RUNTIME_DIR)}; mv {shlex.quote(importing)} {shlex.quote(EVK_RUNTIME_DIR)}; "
        "fi",
        timeout=900,
    )
    logger.event("evk_geniex_runtime_installed", runtime=EVK_RUNTIME_DIR)
    return EVK_RUNTIME_DIR


def _deploy_geniex_service(
    connection: EvkConnection,
    payload: Payload,
    logger: InstallLogger,
) -> None:
    runtime_root = _deploy_runtime(connection, payload, logger)

    archive = _validated_payload_file(payload, EVK_SPEED_ARTIFACT_ID, logger)
    digest = str(payload.spec(EVK_SPEED_ARTIFACT_ID)["sha256"]).lower()
    cache_root = "/home/ubuntu/qai-conveyor/cache"
    model_root = "/home/ubuntu/qai-conveyor/models"
    remote_archive = f"{cache_root}/{digest}.tar.gz"
    bundle = f"{model_root}/{EVK_SPEED_BUNDLE_NAME}"
    marker = f"{bundle}/.qai-conveyor-archive-sha256"
    connection.run(f"mkdir -p {cache_root} {model_root}")
    remote_hash = connection.run(
        f"if [ -f {shlex.quote(remote_archive)} ]; then sha256sum {shlex.quote(remote_archive)} | cut -d' ' -f1; fi",
        check=False,
    ).strip()
    if remote_hash != digest:
        temporary = remote_archive + ".uploading"
        _upload(connection, archive, temporary, logger)
        connection.run(
            f"test \"$(sha256sum {shlex.quote(temporary)} | cut -d' ' -f1)\" = {shlex.quote(digest)}; "
            f"mv {shlex.quote(temporary)} {shlex.quote(remote_archive)}"
        )

    importing = bundle + ".importing"
    connection.run(
        "set -e; "
        f"if [ \"$(cat {shlex.quote(marker)} 2>/dev/null || true)\" != {shlex.quote(digest)} ]; then "
        f"rm -rf -- {shlex.quote(importing)}; mkdir -p {shlex.quote(importing)}; "
        f"tar -xzf {shlex.quote(remote_archive)} --strip-components=1 -C {shlex.quote(importing)}; "
        f"test -r {shlex.quote(importing + '/geniex_compat.json')}; "
        f"for f in vision_encoder.bin part1_of_4.bin part2_of_4.bin part3_of_4.bin part4_of_4.bin; do test -f {shlex.quote(importing)}/\"$f\"; done; "
        f"printf '%s\\n' {shlex.quote(digest)} >{shlex.quote(importing + '/.qai-conveyor-archive-sha256')}; "
        f"rm -rf -- {shlex.quote(bundle)}; mv {shlex.quote(importing)} {shlex.quote(bundle)}; "
        "fi",
        timeout=900,
    )
    logger.event("evk_speed_bundle_installed", release=EVK_SPEED_RELEASE_ID, bundle=bundle)

    output = connection.run(
        _model_import_command(runtime_root, EVK_SPEED_DATA_DIR, bundle, digest),
        timeout=900,
    )
    logger.event("evk_speed_model_imported", output_tail=output[-2000:])

    encoded_unit = base64.b64encode(
        _geniex_systemd_unit(runtime_root, EVK_SPEED_DATA_DIR).encode("utf-8")
    ).decode("ascii")
    connection.run(
        "set -e; "
        f"printf %s {shlex.quote(encoded_unit)} | base64 -d >/tmp/qai-conveyor-geniex.service; "
        "sudo -n install -m 0644 /tmp/qai-conveyor-geniex.service /etc/systemd/system/qai-conveyor-geniex.service; "
        "rm -f /tmp/qai-conveyor-geniex.service; sudo -n systemctl daemon-reload; "
        "sudo -n systemctl enable --now qai-conveyor-geniex.service",
        timeout=60,
    )
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        state = connection.run(
            f"curl -sf --max-time 4 http://127.0.0.1:{EVK_PORT}/v1/models || true; "
            "printf ' workers='; pgrep -xc geniex-grammar || true",
            timeout=12,
            check=False,
        )
        if EVK_MODEL_ID in state and "workers=1" in state:
            logger.event(
                "evk_speed_geniex_service_ready",
                host=connection.host,
                url=f"http://{connection.host}:{EVK_PORT}",
                model=EVK_MODEL_ID,
                prompt_profile="speed-v1",
                image_size="448x256",
            )
            return
        time.sleep(2.0)
    tail = connection.run(
        "sudo -n systemctl --no-pager --full status qai-conveyor-geniex.service; "
        "sudo -n journalctl -u qai-conveyor-geniex.service -n 120 --no-pager",
        check=False,
    )
    raise TimeoutError(f"EVK GenieX service did not become ready within 180 seconds:\n{tail}")


def ensure_evk(
    payload: Payload,
    logger: InstallLogger,
    explicit_host: str | None = None,
    required: bool = False,
) -> str | None:
    if EVK_SPEED_ARTIFACT_ID not in payload.artifacts:
        raise RuntimeError(
            f"Release payload is missing required GenieX artifact {EVK_SPEED_ARTIFACT_ID}"
        )
    username = payload.credentials.get("evk_username", "ubuntu")
    password = payload.credentials.get("evk_password", "ubuntu")
    logger.add_secrets((password,))
    candidates = discover_ssh_candidates(logger, explicit_host)
    connection = connect_evk(candidates, username, password, logger)
    if connection is None:
        message = "No compatible IQ-9075/QCS9075 EVK was reachable over LAN or USB Ethernet SSH"
        if required:
            raise ConnectionError(message)
        logger.event("evk_not_found", level="WARNING", message=message)
        return None
    try:
        _deploy_geniex_service(connection, payload, logger)
        if direct_model_ready(connection.host):
            url = f"http://{connection.host}:{EVK_PORT}"
            logger.event("evk_service_ready", url=url, release=EVK_SPEED_RELEASE_ID)
            return url
        raise ConnectionError(
            f"EVK GenieX is ready through SSH but {connection.host}:{EVK_PORT} "
            "is not reachable from the host; check the EVK/network firewall"
        )
    finally:
        connection.close()
