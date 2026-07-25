#!/usr/bin/env python3
"""Classify the four warehouse events through a persistent GenieX server.

This client implements the measured two-stage closed-set profile:

1. distinguish a forklift scene from worker-only activity;
2. distinguish marker contact from human avoidance, or box pickup from
   multi-worker aisle motion.

The service must be started with full NPU placement and
``MTMD_VIDEO_FPS=2``. The classifier is intentionally task-specific; it is
not a general Cosmos-Reason2 evaluator.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


BROAD_PROMPT = (
    "Which broad scene is directly visible? Answer one letter only: "
    "A) a forklift is prominent in the scene; "
    "B) no forklift is prominent and the visible event is worker activity."
)
FORKLIFT_PROMPT = (
    "Which event is directly visible? Answer one letter only: "
    "A) a forklift pushes or knocks down a striped safety marker; "
    "B) a worker dodges or moves away from an approaching forklift."
)
WORKER_PROMPT = (
    "Does a worker visibly bend down to pick up or carry a cardboard box "
    "in this video? Answer one letter only: A) yes; B) no."
)
AB_GRAMMAR = "root ::= [AB]"

UrlOpen = Callable[..., Any]


def _completion_url(server_url: str) -> str:
    return f"{server_url.rstrip('/')}/v1/chat/completions"


def _request_letter(
    *,
    server_url: str,
    model: str,
    video_url: str,
    prompt: str,
    timeout_seconds: float,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": video_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_completion_tokens": 4,
        "enable_think": False,
        "top_k": 1,
        "temperature": 0,
        "seed": 42,
        "grammar_string": AB_GRAMMAR,
    }
    request = urllib.request.Request(
        _completion_url(server_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            # The measured soak uses independent connections. An interrupted
            # keep-alive client can otherwise leave this experimental server
            # waiting on an abandoned video request.
            "Connection": "close",
        },
        method="POST",
    )
    started = time.monotonic()
    with urlopen(request, timeout=timeout_seconds) as response:
        result = json.load(response)
    elapsed = time.monotonic() - started

    try:
        letter = result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError("GenieX response has no assistant content") from exc
    if letter not in {"A", "B"}:
        raise ValueError(f"GenieX returned an out-of-grammar answer: {letter!r}")
    return letter, elapsed


def classify_video(
    *,
    server_url: str,
    model: str = "local/cosmos-reason2-2b:Q4_0",
    video_url: str,
    timeout_seconds: float = 30.0,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, Any]:
    broad, broad_seconds = _request_letter(
        server_url=server_url,
        model=model,
        video_url=video_url,
        prompt=BROAD_PROMPT,
        timeout_seconds=timeout_seconds,
        urlopen=urlopen,
    )
    detail_prompt = FORKLIFT_PROMPT if broad == "A" else WORKER_PROMPT
    detail, detail_seconds = _request_letter(
        server_url=server_url,
        model=model,
        video_url=video_url,
        prompt=detail_prompt,
        timeout_seconds=timeout_seconds,
        urlopen=urlopen,
    )

    if broad == "A":
        event = "forklift_marker_knockdown" if detail == "A" else "forklift_human_near_miss"
    else:
        event = "routine_box_pickup" if detail == "A" else "multi_worker_aisle_exit"

    return {
        "event": event,
        "broad_answer": broad,
        "detail_answer": detail,
        "request_seconds": [round(broad_seconds, 6), round(detail_seconds, 6)],
        "total_seconds": round(broad_seconds + detail_seconds, 6),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--video",
        type=Path,
        help="Video file visible to the GenieX server; intended for same-EVK use.",
    )
    source.add_argument(
        "--video-url",
        help="Video URL passed verbatim to GenieX, for example file:///path/clip.mp4.",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:18181")
    parser.add_argument("--model", default="local/cosmos-reason2-2b:Q4_0")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")

    if args.video is not None:
        video_path = args.video.expanduser().resolve()
        if not video_path.is_file():
            raise SystemExit(f"Video does not exist: {video_path}")
        video_url = video_path.as_uri()
    else:
        video_url = args.video_url

    try:
        result = classify_video(
            server_url=args.server_url,
            model=args.model,
            video_url=video_url,
            timeout_seconds=args.timeout_seconds,
        )
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise SystemExit(f"Classification failed: {exc}") from exc

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
