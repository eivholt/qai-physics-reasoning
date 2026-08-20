from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import time
import urllib.request
from pathlib import Path

from PIL import Image

from common import per_class_metrics
from generative_common import (
    COMPACT_USER_PROMPT,
    EVK_SINGLE_LANE_USER_PROMPT,
    EVK_SPEED_CLASSIFIER_USER_PROMPT,
    ORIGINAL_USER_PROMPT,
    SYSTEM_PROMPT,
    discover_samples,
    prediction_from_response,
)


def post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def post_stream(url: str, payload: dict, timeout: float) -> tuple[dict, float | None]:
    """Collect an OpenAI SSE response and return it in non-streaming shape."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_ms: float | None = None
    chunks: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            token = (choice.get("delta") or {}).get("content", "")
            if not token:
                token = (choice.get("message") or {}).get("content", "")
            if token:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000.0
                chunks.append(token)
    return {
        "choices": [{"message": {"content": "".join(chunks)}}],
        "usage": usage,
    }, first_token_ms


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an OpenAI-compatible Reason2 adapter endpoint."
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="Cosmos-Reason2-2B-Parcel-FT")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument(
        "--prompt-profile",
        choices=("original", "compact", "evk-single-lane-v1", "evk-speed-v1"),
        default="original",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the selected balanced panel for reliability/thermal soak testing.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use OpenAI SSE streaming and record time to the first output token.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=0,
        help="Resize every image losslessly to this width before submission (0 keeps source size).",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=0,
        help="Resize every image losslessly to this height before submission (0 keeps source size).",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    prompt = {
        "original": ORIGINAL_USER_PROMPT,
        "compact": COMPACT_USER_PROMPT,
        "evk-single-lane-v1": EVK_SINGLE_LANE_USER_PROMPT,
        "evk-speed-v1": EVK_SPEED_CLASSIFIER_USER_PROMPT,
    }[args.prompt_profile]
    samples = discover_samples(Path(args.dataset), args.split)
    if args.limit_per_class:
        counts: dict[str, int] = {}
        selected = []
        for sample in samples:
            code = sample["code"]
            if counts.get(code, 0) < args.limit_per_class:
                selected.append(sample)
                counts[code] = counts.get(code, 0) + 1
        samples = selected
    if not samples:
        raise SystemExit("No evaluation samples")
    selected_count = len(samples)
    samples = [
        {**sample, "soak_pass": pass_index + 1}
        for pass_index in range(args.repeat)
        for sample in samples
    ]

    rows = []
    for index, sample in enumerate(samples, 1):
        mime = "image/png"
        image_path = Path(sample["image"])
        if bool(args.image_width) != bool(args.image_height):
            raise ValueError("--image-width and --image-height must be set together")
        if args.image_width and args.image_height:
            with Image.open(image_path) as source:
                resized = source.convert("RGB").resize(
                    (args.image_width, args.image_height), Image.Resampling.LANCZOS
                )
                buffer = io.BytesIO()
                resized.save(buffer, format="PNG", optimize=False)
                image_bytes = buffer.getvalue()
        else:
            image_bytes = image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        user_message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                {"type": "text", "text": prompt},
            ],
        }
        messages = [user_message] if args.prompt_profile in {
            "evk-single-lane-v1",
            "evk-speed-v1",
        } else [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            user_message,
        ]
        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": 0,
            "max_completion_tokens": (
                1 if args.prompt_profile == "evk-speed-v1" else args.max_new_tokens
            ),
            "stream": args.stream,
        }
        started = time.perf_counter()
        request_url = args.endpoint.rstrip("/") + "/v1/chat/completions"
        if args.stream:
            response, ttft_ms = post_stream(request_url, payload, args.timeout)
        else:
            response = post_json(request_url, payload, args.timeout)
            ttft_ms = None
        wall_ms = (time.perf_counter() - started) * 1000.0
        answer = response["choices"][0]["message"]["content"]
        prediction, parsed = prediction_from_response(answer)
        usage = response.get("usage") or {}
        rows.append(
            {
                **sample,
                "truth": sample["label"],
                "prediction": prediction,
                "answer": answer,
                "parsed": parsed,
                "wall_ms": wall_ms,
                "ttft_ms": ttft_ms,
                "decode_after_ttft_ms": (
                    max(0.0, wall_ms - ttft_ms) if ttft_ms is not None else None
                ),
                "server_ms": response.get("fine_tuned_latency_ms"),
                "usage": usage,
            }
        )
        print(
            f"[{index}/{len(samples)}] {sample['label']} -> "
            f"{prediction or 'INVALID'} wall={wall_ms:.0f}ms "
            f"tokens={usage.get('prompt_tokens', '?')}+"
            f"{usage.get('completion_tokens', '?')}",
            flush=True,
        )

    report = per_class_metrics(rows)
    server_times = [
        float(row["server_ms"])
        for row in rows
        if row.get("server_ms") is not None
    ]
    wall_times = [float(row["wall_ms"]) for row in rows]
    ttft_times = [float(row["ttft_ms"]) for row in rows if row["ttft_ms"] is not None]
    decode_times = [
        float(row["decode_after_ttft_ms"])
        for row in rows
        if row["decode_after_ttft_ms"] is not None
    ]
    report.update(
        {
            "endpoint": args.endpoint,
            "model": args.model,
            "dataset": args.dataset,
            "split": args.split,
            "prompt_profile": args.prompt_profile,
            "prompt": prompt,
            "image_width": args.image_width or None,
            "image_height": args.image_height or None,
            "stream": args.stream,
            "repeat": args.repeat,
            "selected_count": selected_count,
            "mean_server_ms": statistics.fmean(server_times) if server_times else None,
            "median_server_ms": statistics.median(server_times) if server_times else None,
            "mean_latency_ms": statistics.fmean(wall_times) if wall_times else None,
            "p95_latency_ms": percentile(wall_times, 0.95),
            "p99_latency_ms": percentile(wall_times, 0.99),
            "cold_start_latency_ms": wall_times[0] if wall_times else None,
            "mean_warm_latency_ms": (
                statistics.fmean(wall_times[1:]) if len(wall_times) > 1 else None
            ),
            "mean_ttft_ms": statistics.fmean(ttft_times) if ttft_times else None,
            "mean_warm_ttft_ms": (
                statistics.fmean(ttft_times[1:]) if len(ttft_times) > 1 else None
            ),
            "mean_decode_after_ttft_ms": (
                statistics.fmean(decode_times) if decode_times else None
            ),
            "rows": rows,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "accuracy": report["accuracy"],
                "per_class_recall": report["per_class_recall"],
                "mean_server_ms": report["mean_server_ms"],
                "median_server_ms": report["median_server_ms"],
                "mean_latency_ms": report["mean_latency_ms"],
                "mean_warm_latency_ms": report["mean_warm_latency_ms"],
                "p95_latency_ms": report["p95_latency_ms"],
                "p99_latency_ms": report["p99_latency_ms"],
                "mean_warm_ttft_ms": report["mean_warm_ttft_ms"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
