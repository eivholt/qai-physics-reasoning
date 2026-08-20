#!/usr/bin/env python3
"""Validate the resident EVK Reason2 service on prepared lossless observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import statistics
import time
from urllib.request import Request, urlopen

from generative_common import (
    COMPACT_USER_PROMPT,
    EVK_FLAT_OVERHANG_USER_PROMPT,
    EVK_SINGLE_LANE_USER_PROMPT,
    EVK_SPEED_CLASSIFIER_USER_PROMPT,
    ORIGINAL_USER_PROMPT,
)


EXPECTED = {"G": "SUPPORTED", "A": "OVER_EDGE", "R": "FALLEN"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://192.168.1.158:18183")
    parser.add_argument(
        "--model",
        default="local/cosmos-reason2-2b",
        help="OpenAI-compatible model id exposed by the target EVK service",
    )
    parser.add_argument(
        "--media-root",
        default="/home/ubuntu/qai-conveyor/run/media/validation-30",
        help="EVK-local directory containing G, A, and R subdirectories",
    )
    parser.add_argument("--count-per-class", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=45001,
        help="dataset filename seed shared by G, A, and R",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=(
            "original",
            "compact",
            "evk-single-lane-v1",
            "evk-flat-overhang-v2",
            "evk-speed-v1",
        ),
        default="evk-single-lane-v1",
    )
    parser.add_argument(
        "--media-extension",
        choices=("mp4", "png"),
        default="mp4",
        help="extension of the already-uploaded EVK-local observations",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Measure first-token latency from the OpenAI SSE stream.",
    )
    return parser.parse_args()


def prediction_from_content(content: str) -> str:
    closed_set = content.strip().upper()
    if closed_set in EXPECTED:
        return EXPECTED[closed_set]
    value = json.loads(content)
    fallen = str(value.get("fallen_test", "")).strip().upper()
    unstable = str(value.get("unstable_test", "")).strip().upper()
    if fallen == "YES":
        return "FALLEN"
    if fallen == "NO" and unstable == "YES":
        return "OVER_EDGE"
    if fallen == "NO" and unstable == "NO":
        return "SUPPORTED"
    return "INVALID"


def infer(
    endpoint: str,
    model: str,
    media: PurePosixPath,
    timeout: float,
    prompt: str,
    prompt_profile: str,
    stream: bool,
) -> tuple[str, float, dict[str, float]]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{media}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "top_k": 1,
        "seed": 42,
        "enable_think": False,
        "max_completion_tokens": 1 if prompt_profile == "evk-speed-v1" else 13,
        "stream": stream,
    }
    request = Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urlopen(request, timeout=timeout) as response:
        if stream:
            chunks: list[str] = []
            first_token_ms: float | None = None
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                choices = event.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})
                token = delta.get("content", "")
                if not token:
                    token = choice.get("message", {}).get("content", "")
                if token:
                    if first_token_ms is None:
                        first_token_ms = (time.monotonic() - started) * 1000.0
                    chunks.append(token)
            content = "".join(chunks)
            result: dict[str, object] = {}
        else:
            result = json.load(response)
            content = result["choices"][0]["message"]["content"]
    latency_ms = (time.monotonic() - started) * 1000.0
    service_timing = {
        key: float(result[key])
        for key in ("evk_inference_ms", "evk_prepare_ms", "evk_execute_ms")
        if key in result
    }
    if stream:
        service_timing["ttft_ms"] = first_token_ms if first_token_ms is not None else latency_ms
        service_timing["decode_after_ttft_ms"] = max(
            0.0, latency_ms - service_timing["ttft_ms"]
        )
    return content, latency_ms, service_timing


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    prompt = {
        "original": ORIGINAL_USER_PROMPT,
        "compact": COMPACT_USER_PROMPT,
        "evk-single-lane-v1": EVK_SINGLE_LANE_USER_PROMPT,
        "evk-flat-overhang-v2": EVK_FLAT_OVERHANG_USER_PROMPT,
        "evk-speed-v1": EVK_SPEED_CLASSIFIER_USER_PROMPT,
    }[args.prompt_profile]
    confusion = {
        expected: {prediction: 0 for prediction in (*EXPECTED.values(), "INVALID", "ERROR")}
        for expected in EXPECTED.values()
    }
    remote_root = PurePosixPath(args.media_root)
    for code, expected in EXPECTED.items():
        for index in range(1, args.count_per_class + 1):
            name = f"{code}-s{args.seed}-{index:06d}.{args.media_extension}"
            media = remote_root / code / name
            content: str | None = None
            try:
                content, latency_ms, service_timing = infer(
                    args.endpoint,
                    args.model,
                    media,
                    args.timeout,
                    prompt,
                    args.prompt_profile,
                    args.stream,
                )
                prediction = prediction_from_content(content)
                row: dict[str, object] = {
                    "image": name,
                    "expected": expected,
                    "prediction": prediction,
                    "latency_ms": round(latency_ms, 3),
                    **service_timing,
                    "content": content,
                }
            except Exception as error:  # Keep the full balanced run diagnostic.
                prediction = "ERROR"
                row = {
                    "image": name,
                    "expected": expected,
                    "prediction": prediction,
                    "error": repr(error),
                    **({"content": content} if content is not None else {}),
                }
            confusion[expected][prediction] += 1
            rows.append(row)
            print(f"{name}: {prediction} ({row.get('latency_ms', 'error')} ms)", flush=True)

    count = len(rows)
    correct = sum(row["expected"] == row["prediction"] for row in rows)
    latencies = [float(row["latency_ms"]) for row in rows if "latency_ms" in row]
    prepare_latencies = [float(row["evk_prepare_ms"]) for row in rows if "evk_prepare_ms" in row]
    execute_latencies = [float(row["evk_execute_ms"]) for row in rows if "evk_execute_ms" in row]
    ttft_latencies = [float(row["ttft_ms"]) for row in rows if "ttft_ms" in row]
    decode_latencies = [
        float(row["decode_after_ttft_ms"])
        for row in rows
        if "decode_after_ttft_ms" in row
    ]
    result = {
        "count": count,
        "endpoint": args.endpoint,
        "model": args.model,
        "media_root": str(args.media_root),
        "media_extension": args.media_extension,
        "dataset_seed": args.seed,
        "prompt_profile": args.prompt_profile,
        "prompt": prompt,
        "stream": args.stream,
        "accuracy": correct / count if count else 0.0,
        "per_class_recall": {
            expected: confusion[expected][expected] / args.count_per_class
            for expected in EXPECTED.values()
        },
        "mean_latency_ms": statistics.fmean(latencies) if latencies else None,
        "cold_start_latency_ms": latencies[0] if latencies else None,
        "mean_warm_latency_ms": (
            statistics.fmean(latencies[1:]) if len(latencies) > 1 else None
        ),
        "mean_prepare_ms": statistics.fmean(prepare_latencies) if prepare_latencies else None,
        "mean_execute_ms": statistics.fmean(execute_latencies) if execute_latencies else None,
        "mean_ttft_ms": statistics.fmean(ttft_latencies) if ttft_latencies else None,
        "mean_warm_ttft_ms": (
            statistics.fmean(ttft_latencies[1:]) if len(ttft_latencies) > 1 else None
        ),
        "mean_decode_after_ttft_ms": statistics.fmean(decode_latencies) if decode_latencies else None,
        "confusion_matrix": confusion,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in result if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
