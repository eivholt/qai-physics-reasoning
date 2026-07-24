# NVIDIA PhysicalAI SDG-Warehouse preview benchmark

This package defines a small, reproducible video-reasoning benchmark from
NVIDIA's official
[PhysicalAI SDG-Warehouse dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes).
It deliberately downloads only the dataset-card previews (about 24 MB), not
the multi-terabyte WebDataset shards.

The source is version `v1.0`, revision
`d5b88d3abcf659f304a107f4336b71b4e2159133`, under
[OpenMDW-1.1](https://openmdw.ai/license/). NVIDIA says the scenes were
rendered in Isaac Sim with Isaac Replicator Object and Isaac Replicator Agent.
The four labels are known from the staged scenario definitions:

- forklift–human near-miss;
- forklift–shelf collision;
- warehouse fire and worker evacuation; and
- routine box pickup, the non-incident negative control.

`benchmark.json` pins every input URL, byte count, and SHA-256 digest. A hash
mismatch is an error, including if the upstream file ever changes.

## Fetch the lightweight media

From the repository root:

```bash
python scripts/fetch_nvidia_sdg_warehouse.py
```

This anonymously downloads the four animated WebP previews and four scenario
JPGs into the ignored `artifacts/nvidia_sdg_warehouse/` directory. The helper
does not import `huggingface_hub`, inspect environment tokens, or send an
authorization header.

To download only the animated previews and extract the selected temporal
frames:

```bash
python scripts/fetch_nvidia_sdg_warehouse.py \
  --asset-set clips \
  --extract-rgb
```

Pillow is required only for extraction:

```bash
python -m pip install Pillow
```

The generated files are:

```text
artifacts/nvidia_sdg_warehouse/
├── assets/
│   ├── clip_nearmiss.webp
│   ├── clip_forklift_collision.webp
│   ├── clip_fire.webp
│   └── clip_box_pickup.webp
└── frames/
    ├── predict_near_miss/
    │   ├── frame_0030.png
    │   └── frame_0045.png
    ├── predict_shelf_collision/
    │   ├── frame_0066.png
    │   └── frame_0088.png
    ├── observe_near_miss_avoidance/
    │   ├── frame_0045.png
    │   └── frame_0075.png
    ├── observe_barrier_collision/
    │   ├── frame_0198.png
    │   └── frame_0225.png
    ├── observe_routine_box_pickup/
    │   ├── frame_0060.png
    │   └── frame_0120.png
    ├── video_barrier_knockdown_4fps/
    │   └── 6 frames: 0198, 0202, 0213, 0217, 0221, 0225
    ├── video_routine_box_pickup_4fps/
    │   └── 6 frames: 0081, 0085, 0095, 0099, 0102, 0106
    ├── video_near_miss_avoidance_4fps/
    │   └── 6 frames: 0050, 0054, 0065, 0069, 0073, 0077
    └── video_fire_evacuation_4fps/
        └── 6 frames: 0107, 0111, 0121, 0125, 0128, 0132
```

Use `--case predict_near_miss` to extract only one paired-frame case. Use
`--case video_barrier_knockdown_4fps` to extract one complete six-frame
sequence. The extraction helper flattens the three ordered temporal pairs while
preserving the frame numbers recorded in the manifest. Use
`--force` only when intentionally replacing a local asset; the replacement is
still accepted only if its digest and byte count match the manifest.

### GPU reference baseline

Run the native upstream Qwen3-VL video path in deterministic BF16 on a CUDA
GPU, selecting the shared normal/shuffled choice controls:

```bash
python scripts/run_video_gpu_baseline.py \
  --checkpoint /path/to/Cosmos-Reason2-2B \
  --frame-root artifacts/nvidia_sdg_warehouse/frames \
  --output artifacts/nvidia_sdg_warehouse/gpu-bf16 \
  --case video_barrier_knockdown_4fps \
  --case video_routine_box_pickup_4fps \
  --case video_near_miss_avoidance_4fps \
  --case video_fire_evacuation_4fps \
  --prompt four_scene_event_choice \
  --prompt four_scene_event_choice_shuffled
```

The runner fails closed if local tokenization exceeds the pinned CL512/AR128
prefill-safe limit or differs from a manifest budget. Output JSON contains
relative frame names and hashes; local absolute paths are omitted unless
`--include-local-paths` is explicitly supplied.

To reproduce the wording ablation, replace the two `--prompt` selectors with
`four_scene_compact_choice` and `four_scene_compact_choice_shuffled`.
For the focused box-versus-near-miss diagnostic, select only those two cases
and use `box_near_pairwise_choice` plus
`box_near_pairwise_choice_shuffled`.

## Preview layout

The WebPs are animated contact sheets, not ordinary RGB videos. Every
1920x216 frame concatenates five 384x216 panels:

```text
RGB | depth | instance segmentation | shaded segmentation | canny edges
```

The helper crops only the leftmost `x=0..383` RGB panel. Passing the complete
1920-pixel frame to a vision model would mix rendered annotations into the
visual input and invalidate the intended RGB benchmark.

## Tasks and scoring

The manifest defines four event-classification cases, two predictive
two-frame cases, three paired-frame observation diagnostics, and four
three-pair video diagnostics:

- `predict_near_miss` uses preview frames 30 and 45. A correct answer identifies
  the converging forklift/worker paths or collision risk and predicts a dodge
  or near-miss rather than asserting that contact definitely occurs.
- `predict_shelf_collision` uses frames 66 and 88, conservatively ahead of the
  visible impact. The official scenario label is shelf collision, while this
  preview angle most clearly exposes a yellow-and-black striped marker or shelf
  guard. A correct answer predicts contact with either the official scenario
  object or that concrete visible target and the resulting knock-over risk.
- `observe_near_miss_avoidance` uses frames 45 and 75, revealing the worker's
  movement away from the forklift. This separates recognizing the staged
  avoidance from predicting it before it happens.
- `observe_barrier_collision` uses frames 198 and 225, where an orange striped
  safety marker changes from upright to knocked over beside the forklift. This
  is a concrete, visible temporal-change control.
- `observe_routine_box_pickup` uses frames 60 and 120. The worker approaches
  and then carries a box without an accident, making this the required
  non-incident control for detecting blanket hazard answers.
- `video_barrier_knockdown_4fps` uses three timestamped temporal pairs spanning
  frames 198–225. The orange striped marker starts upright, rotates after
  forklift contact, and ends on the floor. A passing answer must identify both
  the state change and the forklift as its cause.
- `video_routine_box_pickup_4fps` uses three timestamped temporal pairs spanning
  frames 81–106. The worker rises with the cardboard box while the nearby
  forklift remains stationary. A passing answer must identify the pickup and
  explicitly say that no collision or accident occurs.
- `video_near_miss_avoidance_4fps` uses three pairs spanning frames 50–77. The
  worker moves clear of the approaching forklift without contact. A passing
  answer must identify the worker's avoidance movement.
- `video_fire_evacuation_4fps` uses three pairs spanning frames 107–132. Two
  workers visibly leave their earlier aisle positions. Scoring requires only
  that directly visible multi-worker motion; recognizing that the official
  staged scenario is a warehouse fire is not required because the fire is
  tiny or occluded in the preview RGB panel.

For automated scoring, normalize the answer to lower case and accept a
predictive case only when it conveys at least one `required_concepts_any`
scene/risk idea **and** at least one `outcome_concepts_any` future-outcome
idea, does not convey a `must_not_assert` idea, and matches the expected
incident boolean. Requiring both groups prevents a generic hazard description
from passing a next-event prediction. Treat the concept strings as semantic
rubrics, not exact substrings. Keep the routine box pickup case in every run
to detect a model that labels all warehouse motion as dangerous.

Score each observation diagnostic on its own visible event: it must convey one
`required_concepts_any` idea, none of the `must_not_assert` ideas, and match the
expected incident state. A case with `incident_state_concepts_any` must also
convey one of those ideas; merely omitting an accident claim is not enough for
the negative control. Do not count observation cases as successful future
prediction; their purpose is to locate failures in basic visual grounding or
two-frame change recognition.

Score a three-pair video diagnostic using the `scoring_rule` embedded in its
`expected` object. The barrier case requires one state change and one forklift
cause. The box negative control requires one action and one explicit
non-incident statement. The near-miss case requires avoidance, and the
fire-scenario case requires directly visible multi-worker movement. Every
case rejects its `must_not_assert` concepts.

The shared `video_case_profile` is sized for the current CL512 edge experiment:
three 224x384 pairs produce 84 visual tokens each, or 252 total. The freeform
prompts occupy 313–320 tokens, the original three-choice controls occupy
341–344 tokens, the expanded four-choice controls occupy 357–360 tokens, and
the compact four-choice diagnostic occupies 332–335 tokens. The focused
box-versus-near-miss prompt occupies 327 tokens. All remain below the pinned
384-token CL512/AR128 prefill ceiling. The pair midpoint timestamps in
`benchmark.json` are the exact values passed to the input preparer before its
documented one-decimal prompt formatting.

### Recorded four-scene result

The expanded control uses one four-choice answer set for all scenes and then
shuffles it consistently. Normal order is marker `A`, box `B`, near miss `C`,
workers leaving aisles `D`; shuffled order is workers `A`, near miss `B`,
marker `C`, box `D`.

| Scene | Order | Expected | BF16 GPU | IQ-9075 NPU | NPU TTFT |
|---|---|---:|---:|---:|---:|
| Marker knockdown | Normal | A | A | A | 743.5 ms |
| Marker knockdown | Shuffled | C | C | B | 733.9 ms |
| Routine box pickup | Normal | B | B | C | 731.3 ms |
| Routine box pickup | Shuffled | D | D | B | 742.9 ms |
| Near-miss avoidance | Normal | C | C | C | 750.7 ms |
| Near-miss avoidance | Shuffled | B | B | B | 740.8 ms |
| Workers leaving aisles | Normal | D | C | C | 744.7 ms |
| Workers leaving aisles | Shuffled | A | A | A | 732.0 ms |

BF16 GPU scores 7/8, NPU scores 4/8, and the exact answers agree in 5/8
cases. For the first three unambiguous scenes, GPU scores 6/6 and NPU 3/6.
The near-miss video is the strongest result: both devices follow its label
from `C` to `B`, 2/2. GPU and NPU see byte-identical packed pixels and have
identical prompt counts in every row, so the earlier narrow two-scene 4/4
result does not generalize to broad parity. Exact hashes and timings are in
the
[`r4 evidence report`](../../docs/evidence/iq9075_video_four_scene_parity_r4.json).

The compact-prompt diagnostic removes 25 tokens per case while preserving the
same scenes, pixels, labels, and ordering:

| Scene | Order | Expected | BF16 GPU | IQ-9075 NPU | NPU TTFT |
|---|---|---:|---:|---:|---:|
| Marker knockdown | Normal | A | C | C | 733.6 ms |
| Marker knockdown | Shuffled | C | B | B | 733.8 ms |
| Routine box pickup | Normal | B | B | C | 741.8 ms |
| Routine box pickup | Shuffled | D | D | B | 739.4 ms |
| Near-miss avoidance | Normal | C | C | C | 739.7 ms |
| Near-miss avoidance | Shuffled | B | B | B | 731.1 ms |
| Workers leaving aisles | Normal | D | B | D | 732.6 ms |
| Workers leaving aisles | Shuffled | A | A | A | 734.3 ms |

GPU scores 5/8, NPU remains at 4/8, and exact answer parity remains 5/8.
Shorter wording moves the marker and fire errors instead of closing the gap.
The box video remains the clearest persistent NPU-only failure: GPU passes
both orders while NPU fails both in the primary and compact suites.

Removing the marker and worker-motion distractors gives this focused result:

| Scene | Order | Expected | BF16 GPU | IQ-9075 NPU | NPU TTFT |
|---|---|---:|---:|---:|---:|
| Routine box pickup | Box A / near B | A | A | A | 741.9 ms |
| Routine box pickup | Near A / box B | B | B | A | 732.4 ms |
| Near-miss avoidance | Box A / near B | B | B | B | 739.9 ms |
| Near-miss avoidance | Near A / box B | A | A | A | 741.1 ms |

GPU scores 4/4, NPU scores 3/4, and exact answers agree in 3/4 cases.
The NPU can distinguish box pickup when box is listed first, but it keeps
option `A` after the box label moves to `B`. The failure is therefore
order-sensitive, not complete visual confusion.

The predictive frame times are assumptions, not simulator ground truth. They
are derived from each preview's decoded frame count divided by the scenario
duration in the dataset card. Preview export cadence is not documented, and
the complete shards use 30 fps. Use the official MP4 and structured annotations
from a selected shard when exact timestamps, contact geometry, or
frame-accurate scoring are required.

This synthetic benchmark is useful for model comparison, not as evidence that
a system is safe for warehouse deployment.

The helper keeps all downloaded media under the ignored `artifacts/`
directory, so this repository does not redistribute NVIDIA's dataset
materials. If you redistribute a fetched preview or any derivative that is
covered as Model Materials, review OpenMDW-1.1 and retain the license agreement
and applicable notices with that distribution.
