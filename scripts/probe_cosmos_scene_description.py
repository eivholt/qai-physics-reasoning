#!/usr/bin/env python3
"""Ask Cosmos-Reason2 to describe an Isaac sensor frame or video window.

This is a perception diagnostic, not a control request. It deliberately does
not expose an actuation grammar or apply a result to the live Isaac scene.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_URL = "http://127.0.0.1:18080"
DEFAULT_MODEL = "host/cosmos-reason2-2b:BF16"
DEFAULT_PROMPT = """\
You are inspecting a fixed overhead warehouse safety-camera recording.
Describe only what is visibly supported by the complete image or chronological
video. Do not assume a task, destination, cooperation, or intent that is not
visible.

Write exactly these seven labeled lines, at most 25 words per line:
ACTORS: visible mobile actors, appearance, and actor type
LOCATIONS: positions relative to shelves, pallets, walls, and aisles
MOTION: visible direction across the window, or indeterminate
DISTANCE: decreasing, increasing, unchanged, or indeterminate
PATH_OVERLAP: yes, no, or indeterminate, followed by visible evidence
UNCERTAINTY: anything occluded, small, dark, or ambiguous
SCENE_SUMMARY: one factual sentence
"""
IDENTITY_HINTS = """\

Known visual identity bindings for this scene: the compact low-profile white
unit is RobotBlue, an autonomous mobile robot. The large blue/black/yellow
industrial vehicle is a forklift. These labels identify appearance only; do
not assume either actor's motion, task, destination, or safety state.
"""


def _media_url(media_path: Path, *, absolute_file_url: bool = False) -> str:
    resolved = media_path.resolve()
    if absolute_file_url:
        return resolved.as_uri()
    try:
        relative = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            "Media must be inside the repository so the local model server can "
            "read it through its bounded media root"
        ) from exc
    return "file://" + relative.as_posix()


def request_description(
    *,
    media_path: Path,
    server_url: str,
    model: str,
    prompt: str = DEFAULT_PROMPT,
    identity_hints: bool = False,
    absolute_file_url: bool = False,
    timeout_seconds: float = 120.0,
    max_completion_tokens: int = 256,
) -> dict[str, Any]:
    """Return the exact diagnostic prompt, response, and timing."""
    if not media_path.is_file():
        raise FileNotFoundError(media_path)
    effective_prompt = prompt + (IDENTITY_HINTS if identity_hints else "")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _media_url(
                                media_path,
                                absolute_file_url=absolute_file_url,
                            )
                        },
                    },
                    {"type": "text", "text": effective_prompt},
                ],
            }
        ],
        "max_completion_tokens": max_completion_tokens,
        "enable_think": False,
        "temperature": 0,
        "top_k": 1,
        "repeat_penalty": 1.15,
        "seed": 42,
    }
    request = urllib.request.Request(
        f"{server_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Connection": "close",
        },
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response_payload = json.load(response)
    elapsed = time.perf_counter() - started
    choices = response_payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"Model returned no choices: {response_payload!r}")
    description = choices[0].get("message", {}).get("content")
    if not isinstance(description, str) or not description.strip():
        raise RuntimeError(
            f"Model returned no text description: {response_payload!r}"
        )
    return {
        "schema_version": "1.0",
        "purpose": "perception_diagnostic_only_no_actuation",
        "media": str(media_path.resolve()),
        "media_url": _media_url(
            media_path,
            absolute_file_url=absolute_file_url,
        ),
        "server_url": server_url,
        "model": model,
        "elapsed_seconds": elapsed,
        "identity_hints": identity_hints,
        "prompt": effective_prompt,
        "description": description.strip(),
        "usage": response_payload.get("usage"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Perception-only question sent with the media.",
    )
    parser.add_argument(
        "--identity-hints",
        action="store_true",
        help="Tell the model the known visual appearance of RobotBlue/forklift",
    )
    parser.add_argument(
        "--absolute-file-url",
        action="store_true",
        help=(
            "Send the absolute file URI; use when this diagnostic runs on the "
            "same machine as the model service."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = request_description(
        media_path=args.media,
        server_url=args.server_url,
        model=args.model,
        prompt=args.prompt,
        timeout_seconds=args.timeout_seconds,
        max_completion_tokens=args.max_completion_tokens,
        identity_hints=args.identity_hints,
        absolute_file_url=args.absolute_file_url,
    )
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
