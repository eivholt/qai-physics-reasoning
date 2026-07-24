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
    └── predict_shelf_collision/
        ├── frame_0066.png
        └── frame_0088.png
```

Use `--case predict_near_miss` to extract only one prediction case. Use
`--force` only when intentionally replacing a local asset; the replacement is
still accepted only if its digest and byte count match the manifest.

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

The manifest defines four event-classification cases and two predictive
two-frame cases:

- `predict_near_miss` uses preview frames 30 and 45. A correct answer identifies
  the converging forklift/worker paths or collision risk and predicts a dodge
  or near-miss rather than asserting that contact definitely occurs.
- `predict_shelf_collision` uses frames 66 and 88, conservatively ahead of the
  visible impact. A correct answer predicts forklift contact with the shelf or
  a resulting knock-over/debris risk.

For automated scoring, normalize the answer to lower case and accept a
predictive case only when it conveys at least one `required_concepts_any`
scene/risk idea **and** at least one `outcome_concepts_any` future-outcome
idea, does not convey a `must_not_assert` idea, and matches the expected
incident boolean. Requiring both groups prevents a generic hazard description
from passing a next-event prediction. Treat the concept strings as semantic
rubrics, not exact substrings. Keep the routine box pickup case in every run
to detect a model that labels all warehouse motion as dangerous.

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
