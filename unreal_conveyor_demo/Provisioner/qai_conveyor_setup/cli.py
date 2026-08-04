from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import __version__
from .evk import EVK_PORT, ensure_evk
from .host import (
    HOST_MODEL_NAME,
    ensure_host_models,
    ensure_host_runtime,
    is_cpu_runtime,
    start_host_server,
    write_runtime_config,
)
from .logging_support import InstallLogger, user_data_dir
from .payload import Payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="qai-conveyor-setup",
        description="Install, verify, diagnose, and launch the Unreal Reason2 conveyor demo.",
    )
    result.add_argument("--version", action="version", version=__version__)
    subparsers = result.add_subparsers(dest="command", required=True)
    for name in ("install", "ensure", "launch"):
        command = subparsers.add_parser(name)
        command.add_argument("--app-dir", type=Path)
        command.add_argument("--payload-root", type=Path)
        command.add_argument("--evk-host")
        command.add_argument("--skip-host", action="store_true")
        command.add_argument("--skip-evk", action="store_true")
        command.add_argument("--require-evk", action="store_true")
        command.add_argument("--no-launch", action="store_true")
    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("--app-dir", type=Path)
    diagnose.add_argument("--payload-root", type=Path)
    support = subparsers.add_parser("support-bundle")
    support.add_argument("--reason", default="manual")
    support.add_argument("--app-dir", type=Path)
    support.add_argument("--payload-root", type=Path)
    return result


def _configured_evk_host() -> str | None:
    runtime = user_data_dir() / "runtime.json"
    if not runtime.is_file():
        return None
    try:
        url = str(json.loads(runtime.read_text(encoding="utf-8")).get("evk_server_url", ""))
    except (OSError, json.JSONDecodeError):
        return None
    if "://" in url:
        url = url.split("://", 1)[1]
    return url.split(":", 1)[0] or None


def _find_game(app_dir: Path | None) -> Path:
    roots = [app_dir] if app_dir else []
    roots.extend((Path(sys.executable).resolve().parent, Path(__file__).resolve().parents[3]))
    names = ("QaiConveyor.exe",) if sys.platform == "win32" else ("QaiConveyor",)
    candidates: list[Path] = []
    for root in roots:
        if not root or not root.exists():
            continue
        for name in names:
            candidates.extend(root.glob(name))
            candidates.extend(root.glob(f"Binaries/**/{name}"))
            candidates.extend(root.glob(f"Contents/MacOS/{name}"))
            candidates.extend(root.glob(f"QaiConveyor.app/Contents/MacOS/{name}"))
            candidates.extend(root.glob(f"Game/**/{name}"))
    candidates = [path.resolve() for path in candidates if path.is_file() and path.resolve() != Path(sys.executable).resolve()]
    if not candidates:
        raise FileNotFoundError(f"QaiConveyor game executable not found below: {roots}")
    return candidates[0]


def _launch_game(app_dir: Path | None, logger: InstallLogger) -> None:
    executable = _find_game(app_dir)
    logger.event("game_launch", executable=executable)
    kwargs: dict[str, object] = {"cwd": executable.parent, "close_fds": True}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([str(executable)], **kwargs)  # type: ignore[arg-type]


def _diagnose(logger: InstallLogger) -> None:
    logger.event(
        "diagnostic_state",
        platform=sys.platform,
        executable=sys.executable,
        user_data=user_data_dir(),
        runtime_config=(user_data_dir() / "runtime.json").is_file(),
        host_model=(user_data_dir() / "Models" / HOST_MODEL_NAME).is_file(),
    )
    for command in (["ipconfig", "/all"], ["route", "print"]) if sys.platform == "win32" else (["ifconfig"], ["netstat", "-rn"]):
        try:
            logger.command(command, timeout=15, check=False)
        except OSError as error:
            logger.event("diagnostic_command_missing", level="WARNING", argv=command, error=error)


def run(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logger = InstallLogger(args.command)
    try:
        if args.command == "support-bundle":
            try:
                Payload(logger, root=args.payload_root, app_dir=args.app_dir)
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
                logger.event("support_payload_unavailable", level="WARNING", error=error)
            print(logger.support_bundle(args.reason))
            return 0
        if args.command == "diagnose":
            try:
                Payload(logger, root=args.payload_root, app_dir=args.app_dir)
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
                logger.event("diagnostic_payload_unavailable", level="WARNING", error=error)
            _diagnose(logger)
            print(logger.support_bundle("diagnostic run"))
            return 0

        app_dir = args.app_dir.resolve() if args.app_dir else None
        with logger.phase("load_release_payload"):
            payload = Payload(logger, root=args.payload_root, app_dir=app_dir)

        host_url = "http://127.0.0.1:18080"
        if not args.skip_host:
            with logger.phase("install_host_reason2"):
                model, projector = ensure_host_models(payload, logger)
                runtime = ensure_host_runtime(payload, logger)
                try:
                    host_url = start_host_server(runtime, model, projector, logger)
                except (OSError, RuntimeError, TimeoutError) as error:
                    if sys.platform != "win32" or is_cpu_runtime(runtime):
                        raise
                    logger.event(
                        "host_cuda_fallback",
                        level="WARNING",
                        error=error,
                        message="CUDA host runtime failed; retrying with the portable CPU runtime",
                    )
                    runtime = ensure_host_runtime(payload, logger, force_cpu=True)
                    host_url = start_host_server(runtime, model, projector, logger)
        else:
            logger.event("host_install_skipped", level="WARNING")

        evk_url = None
        if not args.skip_evk:
            configured_evk = args.evk_host or _configured_evk_host()
            if args.command == "launch" and not args.evk_host and not args.require_evk:
                if configured_evk:
                    evk_url = f"http://{configured_evk}:{EVK_PORT}"
                    logger.event(
                        "evk_launch_configuration_reused",
                        host=configured_evk,
                        message="Normal launch does not scan or redeploy the EVK; use ensure to repair it",
                    )
                else:
                    logger.event("evk_launch_unconfigured", level="WARNING")
            else:
                with logger.phase("discover_and_provision_evk"):
                    evk_url = ensure_evk(
                        payload,
                        logger,
                        explicit_host=configured_evk,
                        required=args.require_evk,
                    )
        else:
            logger.event("evk_install_skipped", level="WARNING")

        with logger.phase("write_runtime_configuration"):
            write_runtime_config(host_url, evk_url, logger)

        should_launch = args.command == "launch" or (args.command == "install" and not args.no_launch)
        if should_launch:
            with logger.phase("launch_demo"):
                _launch_game(app_dir, logger)
        logger.event("session_complete")
        return 0
    except BaseException as error:
        logger.record_exception(error)
        try:
            bundle = logger.support_bundle(f"{type(error).__name__}: {error}")
            print(f"Setup failed. Support bundle: {bundle}", file=sys.stderr)
        except BaseException as bundle_error:
            print(f"Setup failed and support bundle creation also failed: {bundle_error}", file=sys.stderr)
        print(f"ERROR: {logger.redact(error)}", file=sys.stderr)
        return 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
