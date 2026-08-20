from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from common import per_class_metrics
from generative_common import (
    EDGE_FIRST_USER_PROMPT,
    EVK_SINGLE_LANE_USER_PROMPT,
    EVK_SPEED_CLASSIFIER_USER_PROMPT,
    ORIGINAL_USER_PROMPT,
    SYSTEM_PROMPT,
    discover_samples,
    prediction_from_response,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--skip-per-class",
        type=int,
        default=0,
        help="Skip the first N discovered samples in each class before applying the limit.",
    )
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=288)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--prompt", default=ORIGINAL_USER_PROMPT)
    parser.add_argument(
        "--edge-first",
        action="store_true",
        help="Evaluate the equivalent state-first streaming response contract.",
    )
    parser.add_argument(
        "--evk-single-lane",
        action="store_true",
        help="Evaluate the exact short two-test prompt deployed on the EVK.",
    )
    parser.add_argument(
        "--evk-speed-classifier",
        action="store_true",
        help="Evaluate the user-only, one-token G/A/R speed-v1 contract.",
    )
    args = parser.parse_args()
    if sum((args.edge_first, args.evk_single_lane, args.evk_speed_classifier)) > 1:
        raise ValueError(
            "--edge-first, --evk-single-lane, and --evk-speed-classifier "
            "are mutually exclusive"
        )
    if args.edge_first:
        if args.prompt != ORIGINAL_USER_PROMPT:
            raise ValueError("--edge-first cannot be combined with --prompt")
        args.prompt = EDGE_FIRST_USER_PROMPT
    if args.evk_single_lane:
        if args.prompt != ORIGINAL_USER_PROMPT:
            raise ValueError("--evk-single-lane cannot be combined with --prompt")
        args.prompt = EVK_SINGLE_LANE_USER_PROMPT
    if args.evk_speed_classifier:
        if args.prompt != ORIGINAL_USER_PROMPT:
            raise ValueError("--evk-speed-classifier cannot be combined with --prompt")
        args.prompt = EVK_SPEED_CLASSIFIER_USER_PROMPT

    samples = discover_samples(Path(args.dataset), args.split)
    if args.skip_per_class:
        kept, counts = [], {}
        for sample in samples:
            code = sample["code"]
            seen = counts.get(code, 0)
            counts[code] = seen + 1
            if seen >= args.skip_per_class:
                kept.append(sample)
        samples = kept
    if args.limit_per_class:
        kept, counts = [], {}
        for sample in samples:
            if counts.get(sample["code"], 0) < args.limit_per_class:
                kept.append(sample)
                counts[sample["code"]] = counts.get(sample["code"], 0) + 1
        samples = kept
    if not samples:
        raise SystemExit("No evaluation samples")

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    if args.adapter:
        # Keep merged-checkpoint evaluation usable in minimal inference
        # environments that intentionally do not install PEFT.
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = []
    for index, sample in enumerate(samples, 1):
        with Image.open(sample["image"]) as source:
            image = source.convert("RGB").resize(
                (args.image_width, args.image_height), Image.Resampling.LANCZOS
            )
        user_message = {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ]}
        user_only = args.evk_single_lane or args.evk_speed_classifier
        messages = [user_message] if user_only else [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            user_message,
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=(1 if args.evk_speed_classifier else args.max_new_tokens),
                do_sample=False,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        answer = processor.decode(
            generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        prediction, parsed = prediction_from_response(answer)
        rows.append({
            **sample,
            "truth": sample["label"],
            "prediction": prediction,
            "answer": answer,
            "parsed": parsed,
            "latency_ms": latency_ms,
        })
        print(
            f"[{index}/{len(samples)}] {sample['label']} -> {prediction or 'INVALID'} "
            f"({latency_ms:.0f} ms)", flush=True
        )

    report = per_class_metrics(rows)
    report.update({
        "model": args.model,
        "adapter": args.adapter,
        "dataset": args.dataset,
        "split": args.split,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "rows": rows,
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "accuracy": report["accuracy"],
        "per_class_recall": report["per_class_recall"],
    }, indent=2))


if __name__ == "__main__":
    main()
