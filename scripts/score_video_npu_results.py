#!/usr/bin/env python3
"""Score GenieX NPU video logs against benchmark truth and BF16 GPU results.

The GPU result files define the evaluated subset of the benchmark. This makes
one invocation work for the frozen 20-choice suite, the P3/P4 balanced
extension, or a smaller diagnostic subset:

    python scripts/score_video_npu_results.py \
        --gpu-results /results/gpu-four-scene-r4 \
        --gpu-results /results/gpu-four-scene-compact-r5 \
        --gpu-results /results/gpu-box-near-pairwise-r6 \
        --npu-results baseline=/results/npu-four-scene-r4 \
        --npu-results baseline=/results/npu-four-scene-compact-r5 \
        --npu-results baseline=/results/npu-box-near-pairwise-r6 \
        --npu-results boundary=/results/npu-boundary \
        --npu-results candidate=/results/npu-candidate \
        --output /results/npu-comparison.json

Repeated entries with the same explicit label are merged into one candidate.
Unlabelled entries remain separate candidates.

Only lightweight JSON and text logs are read. No model runtime is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = (
    REPO_ROOT / "benchmarks" / "nvidia_sdg_warehouse" / "benchmark.json"
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
MODERN_OUTPUT = re.compile(
    r"COSMOS_GENIEX_OUTPUT_BEGIN\s*(.*?)\s*COSMOS_GENIEX_OUTPUT_END",
    re.DOTALL,
)
LEGACY_OUTPUT = re.compile(r"\[BEGIN\]:(.*?)\[END\]", re.DOTALL)
CHOICE_PREFIX = re.compile(r"^\s*([A-Z])(?:\b|\))", re.IGNORECASE)
CHOICE_NAMED = re.compile(
    r"\b(?:answer|choice|option)\s*(?:is|:)?\s*([A-Z])\b",
    re.IGNORECASE,
)
TTFT = re.compile(
    r"TTFT(?:\s*\([^)]*\))?\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
    re.IGNORECASE,
)
DECODE = re.compile(
    r"Decode\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*tokens/s",
    re.IGNORECASE,
)
PROMPT_TOKENS = re.compile(r"Prompt tokens\s*:\s*(\d+)", re.IGNORECASE)
GENERATED_TOKENS = re.compile(
    r"Generated tokens\s*:\s*(\d+)", re.IGNORECASE
)
SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")
MAX_RECORDED_ANSWER_CHARS = 1000


@dataclass(frozen=True, order=True)
class ProbeKey:
    """Stable identity of one benchmark choice probe."""

    case_id: str
    prompt_id: str

    @property
    def filename_prefix(self) -> str:
        return f"{self.case_id}.{self.prompt_id}"


@dataclass(frozen=True)
class ProbeExpectation:
    """Ground-truth metadata for one choice probe."""

    key: ProbeKey
    expected_letter: str
    expected_event_label: str | None
    evaluation_role: str
    option_order_control_id: str
    permutation_index: int

    @property
    def permutation_key(self) -> str:
        return (
            f"{self.option_order_control_id}/p{self.permutation_index}"
        )


@dataclass(frozen=True)
class NpuResultSpec:
    """One NPU result root and whether its label was explicitly supplied."""

    label: str
    path: Path
    explicitly_labeled: bool


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _round_metric(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _answer_record(answer: str | None) -> dict[str, Any]:
    if answer is None:
        return {"text": None, "truncated": False}
    return {
        "text": answer[:MAX_RECORDED_ANSWER_CHARS],
        "truncated": len(answer) > MAX_RECORDED_ANSWER_CHARS,
    }


def extract_choice(answer: str | None) -> str | None:
    """Extract an A-D choice using the same rules as the GPU runner."""

    if answer is None:
        return None
    for pattern in (CHOICE_PREFIX, CHOICE_NAMED):
        match = pattern.search(answer)
        if match:
            selected = match.group(1).upper()
            if selected in {"A", "B", "C", "D"}:
                return selected
    return None


def permutation_index(prompt_id: str) -> int:
    """Map the benchmark's normal/shuffled/numbered ids to P1, P2, ..."""

    numbered = re.search(r"_permutation_(\d+)$", prompt_id)
    if numbered:
        value = int(numbered.group(1))
        if value < 1:
            raise ValueError(f"Invalid permutation in prompt id: {prompt_id}")
        return value
    if prompt_id.endswith("_shuffled"):
        return 2
    return 1


def load_expectations(
    benchmark_path: Path,
) -> tuple[
    dict[ProbeKey, ProbeExpectation],
    dict[str, Any],
    bytes,
]:
    """Load all choice probes declared by a benchmark manifest."""

    raw = benchmark_path.read_bytes()
    manifest = json.loads(raw)
    expectations: dict[ProbeKey, ProbeExpectation] = {}
    for case in manifest.get("video_cases", []):
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every video case must have a non-empty id")
        for probe in case.get("choice_probes", []):
            prompt_id = probe.get("id")
            expected_letter = probe.get("expected_letter")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise ValueError(f"{case_id}: choice probe has no valid id")
            if (
                not isinstance(expected_letter, str)
                or expected_letter.upper() not in {"A", "B", "C", "D"}
            ):
                raise ValueError(
                    f"{case_id}/{prompt_id}: expected_letter must be A-D"
                )
            key = ProbeKey(case_id, prompt_id)
            if key in expectations:
                raise ValueError(
                    f"Duplicate benchmark choice probe: {case_id}/{prompt_id}"
                )
            order_id = probe.get("option_order_control_id", prompt_id)
            if not isinstance(order_id, str) or not order_id:
                raise ValueError(
                    f"{case_id}/{prompt_id}: invalid option order control id"
                )
            role = probe.get("evaluation_role", "choice_probe")
            if not isinstance(role, str) or not role:
                raise ValueError(
                    f"{case_id}/{prompt_id}: invalid evaluation role"
                )
            event_label = probe.get("expected_event_label")
            if event_label is not None and not isinstance(event_label, str):
                raise ValueError(
                    f"{case_id}/{prompt_id}: invalid expected event label"
                )
            expectations[key] = ProbeExpectation(
                key=key,
                expected_letter=expected_letter.upper(),
                expected_event_label=event_label,
                evaluation_role=role,
                option_order_control_id=order_id,
                permutation_index=permutation_index(prompt_id),
            )
    if not expectations:
        raise ValueError("Benchmark contains no video choice probes")
    return expectations, manifest, raw


def _safe_source_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "results"


def _root_aliases(roots: Sequence[Path]) -> list[str]:
    """Create unique, path-free aliases for one or more result roots."""

    bases = [_safe_source_component(root.name) for root in roots]
    totals = {base: bases.count(base) for base in set(bases)}
    occurrences: dict[str, int] = {}
    used: set[str] = set()
    aliases: list[str] = []
    for base in bases:
        occurrences[base] = occurrences.get(base, 0) + 1
        if totals[base] == 1:
            candidate = base
        else:
            candidate = f"{base}-{occurrences[base]}"
        suffix = 2
        unique = candidate
        while unique in used:
            unique = f"{candidate}-{suffix}"
            suffix += 1
        used.add(unique)
        aliases.append(unique)
    return aliases


def _source_name(root: Path, path: Path, root_alias: str | None = None) -> str:
    """Return a useful source name without recording an absolute local path."""

    alias = root_alias or _safe_source_component(root.name)
    relative = "/".join(
        _safe_source_component(part)
        for part in path.relative_to(root).parts
    )
    return f"{alias}/{relative}"


def load_gpu_results(
    roots: Sequence[Path],
    expectations: Mapping[ProbeKey, ProbeExpectation],
) -> dict[ProbeKey, dict[str, Any]]:
    """Load per-probe GPU JSON records from one or more result roots."""

    loaded: dict[ProbeKey, dict[str, Any]] = {}
    for root, root_alias in zip(roots, _root_aliases(roots)):
        if not root.is_dir():
            raise ValueError(f"GPU result directory does not exist: {root}")
        for path in sorted(root.rglob("*.gpu_bf16.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            key = ProbeKey(
                str(payload.get("case_id", "")),
                str(payload.get("prompt_id", "")),
            )
            if key not in expectations:
                raise ValueError(
                    "GPU result is not a choice probe in this benchmark: "
                    f"{key.case_id}/{key.prompt_id}"
                )
            if key in loaded:
                raise ValueError(
                    "Duplicate GPU result for "
                    f"{key.case_id}/{key.prompt_id}"
                )
            expected = expectations[key]
            declared_expected = payload.get("expected", {}).get("letter")
            if (
                isinstance(declared_expected, str)
                and declared_expected.upper() != expected.expected_letter
            ):
                raise ValueError(
                    f"{key.case_id}/{key.prompt_id}: GPU expected letter "
                    f"{declared_expected!r} differs from benchmark "
                    f"{expected.expected_letter!r}"
                )
            answer = payload.get("answer")
            if not isinstance(answer, str):
                answer = None
            assessment = payload.get("rubric_assessment", {})
            selected = assessment.get("selected_letter")
            if not isinstance(selected, str):
                selected = extract_choice(answer)
            else:
                selected = selected.upper()
            if selected not in {"A", "B", "C", "D"}:
                selected = None
            timing = payload.get("timing", {})
            gpu_seconds = timing.get("gpu_seconds")
            if not isinstance(gpu_seconds, (int, float)) or isinstance(
                gpu_seconds, bool
            ):
                gpu_seconds = None
            loaded[key] = {
                "answer": answer,
                "selected_letter": selected,
                "correct": selected == expected.expected_letter,
                "gpu_seconds": (
                    float(gpu_seconds) if gpu_seconds is not None else None
                ),
                "source_file": _source_name(root, path, root_alias),
                "sha256": _sha256_file(path),
            }
    if not loaded:
        raise ValueError("No *.gpu_bf16.json files found in GPU result roots")
    return loaded


def _last_number(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    return float(matches[-1]) if matches else None


def _last_integer(pattern: re.Pattern[str], text: str) -> int | None:
    matches = pattern.findall(text)
    return int(matches[-1]) if matches else None


def parse_genie_log(text: str) -> dict[str, Any]:
    """Parse one GenieX log, accepting current and legacy answer markers."""

    cleaned = ANSI_ESCAPE.sub("", text)
    blocks = MODERN_OUTPUT.findall(cleaned)
    marker_format = "cosmos_geniex"
    if not blocks:
        blocks = LEGACY_OUTPUT.findall(cleaned)
        marker_format = "legacy_begin_end"
    answer = blocks[-1].strip() if blocks else None
    if answer == "":
        answer = None
    selected = extract_choice(answer)
    if not blocks:
        status = "missing_output_marker"
    elif answer is None:
        status = "empty_answer"
    elif selected is None:
        status = "unparsed_choice"
    else:
        status = "ok"
    ttft_ms = _last_number(TTFT, cleaned)
    decode_tokens_per_second = _last_number(DECODE, cleaned)
    return {
        "parse_status": status,
        "answer": answer,
        "selected_letter": selected,
        "answer_block_count": len(blocks),
        "marker_format": marker_format if blocks else None,
        "prompt_tokens": _last_integer(PROMPT_TOKENS, cleaned),
        "generated_tokens": _last_integer(GENERATED_TOKENS, cleaned),
        "ttft_ms": ttft_ms,
        "decode_tokens_per_second": decode_tokens_per_second,
    }


def load_npu_results(
    roots: Path | Sequence[Path],
    universe: Iterable[ProbeKey],
) -> tuple[dict[ProbeKey, dict[str, Any]], list[str]]:
    """Load NPU logs matching the GPU-defined probe universe.

    Multiple roots form one candidate. A probe may appear in at most one root.
    """

    normalized_roots = [roots] if isinstance(roots, Path) else list(roots)
    if not normalized_roots:
        raise ValueError("NPU candidate has no result directories")
    expected_filenames = {
        f"{key.filename_prefix}.log": key for key in universe
    }
    loaded: dict[ProbeKey, dict[str, Any]] = {}
    ignored: list[str] = []
    for root, root_alias in zip(
        normalized_roots, _root_aliases(normalized_roots)
    ):
        if not root.is_dir():
            raise ValueError(f"NPU result directory does not exist: {root}")
        for path in sorted(root.rglob("*.log")):
            key = expected_filenames.get(path.name)
            if key is None:
                ignored.append(_source_name(root, path, root_alias))
                continue
            if key in loaded:
                raise ValueError(
                    "Duplicate NPU result for "
                    f"{key.case_id}/{key.prompt_id} across merged roots"
                )
            text = path.read_text(encoding="utf-8", errors="replace")
            loaded[key] = {
                **parse_genie_log(text),
                "source_file": _source_name(root, path, root_alias),
                "sha256": _sha256_file(path),
            }
    return loaded, ignored


def _numeric_summary(values: Iterable[float | int | None]) -> dict[str, Any]:
    usable = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not usable:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(usable),
        "mean": round(statistics.fmean(usable), 6),
        "median": round(statistics.median(usable), 6),
        "min": round(min(usable), 6),
        "max": round(max(usable), 6),
    }


def _aggregate(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(entries)
    parsed = sum(
        entry["npu"]["selected_letter"] is not None for entry in entries
    )
    correct = sum(entry["npu"]["correct"] is True for entry in entries)
    parity_comparable = sum(
        entry["exact_choice_parity"] is not None for entry in entries
    )
    parity = sum(entry["exact_choice_parity"] is True for entry in entries)
    gpu_correct = sum(entry["gpu"]["correct"] is True for entry in entries)
    retained = sum(
        entry["gpu"]["correct"] is True
        and entry["npu"]["correct"] is True
        for entry in entries
    )
    gpu_incorrect = total - gpu_correct
    npu_recoveries = sum(
        entry["gpu"]["correct"] is False
        and entry["npu"]["correct"] is True
        for entry in entries
    )
    statuses: dict[str, int] = {}
    for entry in entries:
        status = entry["npu"]["parse_status"]
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "total": total,
        "parsed": parsed,
        "missing_or_unparsed": total - parsed,
        "parse_status_counts": dict(sorted(statuses.items())),
        "correct": correct,
        "accuracy": _round_metric(_fraction(correct, total)),
        "accuracy_among_parsed": _round_metric(_fraction(correct, parsed)),
        "exact_choice_parity": parity,
        "exact_choice_parity_total": total,
        "exact_choice_parity_rate": _round_metric(
            _fraction(parity, total)
        ),
        "parity_comparable": parity_comparable,
        "parity_among_comparable": _round_metric(
            _fraction(parity, parity_comparable)
        ),
        "gpu_correct_total": gpu_correct,
        "gpu_correct_retained": retained,
        "gpu_correct_retention_rate": _round_metric(
            _fraction(retained, gpu_correct)
        ),
        "gpu_incorrect_total": gpu_incorrect,
        "npu_correct_on_gpu_incorrect": npu_recoveries,
        "timing": {
            "ttft_ms": _numeric_summary(
                entry["npu"]["ttft_ms"] for entry in entries
            ),
            "decode_tokens_per_second": _numeric_summary(
                entry["npu"]["decode_tokens_per_second"]
                for entry in entries
            ),
            "decode_tokens_per_second_nonzero": _numeric_summary(
                entry["npu"]["decode_tokens_per_second"]
                for entry in entries
                if (
                    entry["npu"]["decode_tokens_per_second"] is not None
                    and entry["npu"]["decode_tokens_per_second"] > 0
                )
            ),
            "generated_tokens": _numeric_summary(
                entry["npu"]["generated_tokens"] for entry in entries
            ),
        },
    }


def _grouped_summary(
    entries: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(str(entry[field]), []).append(entry)
    return {
        group: _aggregate(group_entries)
        for group, group_entries in sorted(groups.items())
    }


def _gpu_summary(
    gpu_results: Mapping[ProbeKey, dict[str, Any]],
) -> dict[str, Any]:
    total = len(gpu_results)
    parsed = sum(
        result["selected_letter"] is not None
        for result in gpu_results.values()
    )
    correct = sum(
        result["correct"] is True for result in gpu_results.values()
    )
    return {
        "total": total,
        "parsed": parsed,
        "correct": correct,
        "accuracy": _round_metric(_fraction(correct, total)),
        "timing": {
            "gpu_seconds": _numeric_summary(
                result["gpu_seconds"] for result in gpu_results.values()
            )
        },
        "sources": [
            {
                "file": result["source_file"],
                "sha256": result["sha256"],
            }
            for _, result in sorted(gpu_results.items())
        ],
    }


def _case_entry(
    expected: ProbeExpectation,
    gpu: Mapping[str, Any],
    npu: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if npu is None:
        npu_record: dict[str, Any] = {
            "parse_status": "missing_log",
            "selected_letter": None,
            "correct": False,
            "answer": _answer_record(None),
            "answer_block_count": 0,
            "marker_format": None,
            "prompt_tokens": None,
            "generated_tokens": None,
            "ttft_ms": None,
            "decode_tokens_per_second": None,
            "source_file": None,
            "sha256": None,
        }
    else:
        npu_record = {
            **npu,
            "answer": _answer_record(npu.get("answer")),
            "correct": (
                npu.get("selected_letter") == expected.expected_letter
            ),
        }
    gpu_selected = gpu.get("selected_letter")
    npu_selected = npu_record["selected_letter"]
    parity = (
        gpu_selected == npu_selected
        if gpu_selected is not None and npu_selected is not None
        else None
    )
    return {
        "case_id": expected.key.case_id,
        "prompt_id": expected.key.prompt_id,
        "expected_letter": expected.expected_letter,
        "expected_event_label": expected.expected_event_label,
        "evaluation_role": expected.evaluation_role,
        "option_order_control_id": expected.option_order_control_id,
        "permutation_index": expected.permutation_index,
        "permutation_key": expected.permutation_key,
        "gpu": {
            **gpu,
            "answer": _answer_record(gpu.get("answer")),
        },
        "npu": npu_record,
        "exact_choice_parity": parity,
        "gpu_correct_retained": (
            npu_record["correct"] is True if gpu["correct"] is True else None
        ),
    }


def _candidate_comparison(
    baseline_name: str,
    baseline_entries: Sequence[dict[str, Any]],
    candidate_name: str,
    candidate_entries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_key = {
        (entry["case_id"], entry["prompt_id"]): entry
        for entry in baseline_entries
    }
    candidate_by_key = {
        (entry["case_id"], entry["prompt_id"]): entry
        for entry in candidate_entries
    }
    if baseline_by_key.keys() != candidate_by_key.keys():
        raise ValueError("Candidate comparison universes differ")
    changes: list[dict[str, Any]] = []
    comparable = 0
    changed = 0
    improved = 0
    regressed = 0
    newly_parsed = 0
    lost_parse = 0
    for key in sorted(baseline_by_key):
        baseline = baseline_by_key[key]
        candidate = candidate_by_key[key]
        before = baseline["npu"]["selected_letter"]
        after = candidate["npu"]["selected_letter"]
        if before is not None and after is not None:
            comparable += 1
        if before is None and after is not None:
            newly_parsed += 1
        if before is not None and after is None:
            lost_parse += 1
        if before == after:
            continue
        changed += 1
        before_correct = baseline["npu"]["correct"] is True
        after_correct = candidate["npu"]["correct"] is True
        if not before_correct and after_correct:
            improved += 1
        if before_correct and not after_correct:
            regressed += 1
        changes.append(
            {
                "case_id": key[0],
                "prompt_id": key[1],
                "expected_letter": baseline["expected_letter"],
                "from_letter": before,
                "to_letter": after,
                "from_correct": before_correct,
                "to_correct": after_correct,
            }
        )
    baseline_correct = sum(
        entry["npu"]["correct"] is True for entry in baseline_entries
    )
    candidate_correct = sum(
        entry["npu"]["correct"] is True for entry in candidate_entries
    )
    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "total": len(baseline_entries),
        "comparable": comparable,
        "changed": changed,
        "unchanged": len(baseline_entries) - changed,
        "improved": improved,
        "regressed": regressed,
        "newly_parsed": newly_parsed,
        "lost_parse": lost_parse,
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "correct_delta": candidate_correct - baseline_correct,
        "changes": changes,
    }


def group_npu_candidate_roots(
    candidates: Sequence[NpuResultSpec | tuple[str, Path]],
) -> list[tuple[str, list[Path]]]:
    """Merge repeated explicit labels and keep unlabeled roots distinct.

    Tuple inputs are treated as explicitly labelled for backwards-compatible
    programmatic use. Inferred labels are disambiguated with ``-2``, ``-3``,
    and so on instead of accidentally merging directories with the same name.
    """

    normalized = [
        (
            item
            if isinstance(item, NpuResultSpec)
            else NpuResultSpec(item[0], item[1], True)
        )
        for item in candidates
    ]
    explicit_labels = {
        item.label for item in normalized if item.explicitly_labeled
    }
    grouped: dict[str, list[Path]] = {}
    used_labels = set(explicit_labels)
    inferred_counts: dict[str, int] = {}
    for item in normalized:
        if item.explicitly_labeled:
            grouped.setdefault(item.label, []).append(item.path)
            continue
        base = item.label
        inferred_counts[base] = inferred_counts.get(base, 0) + 1
        suffix = inferred_counts[base]
        label = base if suffix == 1 else f"{base}-{suffix}"
        while label in used_labels or label in grouped:
            suffix += 1
            inferred_counts[base] = suffix
            label = f"{base}-{suffix}"
        used_labels.add(label)
        grouped[label] = [item.path]
    return list(grouped.items())


def build_report(
    benchmark_path: Path,
    gpu_roots: Sequence[Path],
    npu_candidates: Sequence[NpuResultSpec | tuple[str, Path]],
) -> dict[str, Any]:
    """Build a serializable comparison report."""

    expectations, manifest, benchmark_raw = load_expectations(benchmark_path)
    gpu_results = load_gpu_results(gpu_roots, expectations)
    universe = sorted(gpu_results)
    candidate_reports: dict[str, dict[str, Any]] = {}
    for name, roots in group_npu_candidate_roots(npu_candidates):
        if name in candidate_reports:
            raise ValueError(f"Duplicate NPU candidate label: {name}")
        npu_results, ignored = load_npu_results(roots, universe)
        entries = [
            _case_entry(expectations[key], gpu_results[key], npu_results.get(key))
            for key in universe
        ]
        candidate_reports[name] = {
            "source_root_names": _root_aliases(roots),
            "ignored_log_files": ignored,
            "summary": _aggregate(entries),
            "by_scene": _grouped_summary(entries, "case_id"),
            "by_permutation": _grouped_summary(
                entries, "permutation_key"
            ),
            "by_evaluation_role": _grouped_summary(
                entries, "evaluation_role"
            ),
            "cases": entries,
        }
    comparisons = [
        _candidate_comparison(
            left_name,
            candidate_reports[left_name]["cases"],
            right_name,
            candidate_reports[right_name]["cases"],
        )
        for left_name, right_name in itertools.combinations(
            candidate_reports, 2
        )
    ]
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Compare GenieX NPU choice accuracy with frozen benchmark truth "
            "and deterministic BF16 GPU outputs"
        ),
        "definitions": {
            "evaluation_universe": (
                "Choice probes represented by the supplied GPU per-case JSON "
                "files; all expected letters come from the benchmark manifest."
            ),
            "exact_choice_parity": (
                "The extracted A-D NPU choice equals the extracted A-D GPU "
                "choice. Missing or unparsed choices count against the "
                "total-denominator parity rate."
            ),
            "gpu_correct_retention": (
                "Among cases the GPU answered correctly, the fraction the NPU "
                "also answered correctly."
            ),
            "permutation": (
                "P1 is the normal option order, P2 is the _shuffled probe, "
                "and explicitly numbered probes map to their number."
            ),
        },
        "benchmark": {
            "id": manifest.get("benchmark_id"),
            "file": benchmark_path.name,
            "sha256": _sha256_bytes(benchmark_raw),
            "source_revision": manifest.get("source", {}).get("revision"),
        },
        "gpu_reference": _gpu_summary(gpu_results),
        "candidates": candidate_reports,
        "answer_changes": comparisons,
    }


def _parse_candidate_spec(value: str) -> NpuResultSpec:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        explicitly_labeled = True
    else:
        raw_path = value
        label = Path(value).name
        explicitly_labeled = False
    if not label or not SAFE_LABEL.fullmatch(label):
        raise argparse.ArgumentTypeError(
            "NPU label must contain only letters, digits, dot, dash, or underscore"
        )
    if not raw_path:
        raise argparse.ArgumentTypeError("NPU result path must not be empty")
    return NpuResultSpec(label, Path(raw_path), explicitly_labeled)


def _print_summary(report: Mapping[str, Any]) -> None:
    gpu = report["gpu_reference"]
    print(
        f"GPU: {gpu['correct']}/{gpu['total']} "
        f"({gpu['accuracy']:.1%})"
    )
    for name, candidate in report["candidates"].items():
        summary = candidate["summary"]
        accuracy = summary["accuracy"]
        parity = summary["exact_choice_parity_rate"]
        retention = summary["gpu_correct_retention_rate"]
        mean_ttft = summary["timing"]["ttft_ms"]["mean"]
        parts = [
            f"{name}: NPU {summary['correct']}/{summary['total']}",
            (
                f"parity {summary['exact_choice_parity']}/"
                f"{summary['exact_choice_parity_total']}"
            ),
            (
                f"GPU-correct retention "
                f"{summary['gpu_correct_retained']}/"
                f"{summary['gpu_correct_total']}"
            ),
        ]
        if accuracy is not None:
            parts[0] += f" ({accuracy:.1%})"
        if parity is not None:
            parts[1] += f" ({parity:.1%})"
        if retention is not None:
            parts[2] += f" ({retention:.1%})"
        if mean_ttft is not None:
            parts.append(f"mean TTFT {mean_ttft:.1f} ms")
        print("; ".join(parts))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--gpu-results",
        type=Path,
        action="append",
        required=True,
        help="GPU result directory; repeat for split frozen-suite outputs",
    )
    parser.add_argument(
        "--npu-results",
        type=_parse_candidate_spec,
        action="append",
        required=True,
        metavar="[LABEL=]DIRECTORY",
        help=(
            "NPU result directory and optional report label; repeated entries "
            "with the same explicit LABEL merge into one candidate, while "
            "unlabelled entries remain distinct"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the full JSON report here; otherwise print it to stdout",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Return an error if any candidate has a missing/unparsed choice",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_report(
            args.benchmark,
            args.gpu_results,
            args.npu_results,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _print_summary(report)
        print(f"Wrote {args.output}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.require_complete and any(
        candidate["summary"]["missing_or_unparsed"]
        for candidate in report["candidates"].values()
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
