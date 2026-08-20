#!/usr/bin/env python3
"""Double-clickable Windows entry point for the accepted Reason2 demo release."""

from __future__ import annotations

from pathlib import Path
import sys

from qai_conveyor_setup.cli import run


def default_install_arguments() -> list[str]:
    root = Path(sys.executable).resolve().parent
    return [
        "install",
        "--app-dir",
        str(root / "Game"),
        "--payload-root",
        str(root / "Payload"),
        "--evk-host",
        "192.168.1.158",
    ]


def main() -> None:
    arguments = sys.argv[1:] or default_install_arguments()
    raise SystemExit(run(arguments))


if __name__ == "__main__":
    main()
