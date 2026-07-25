# Deploy Cosmos-Reason2-2B on the IQ-9075 NPU

This is the shortest verified path from NVIDIA's Hugging Face checkpoint to
image and native-video inference on a Dragonwing IQ-9075 EVK. It exports a
standard Q4_0 GGUF plus an F16 multimodal projector, builds a patched Qualcomm
GenieX v0.3.17 runtime, and runs the vision tower and language decoder with
`--compute npu`.

The result is an experimental deployment, not a Qualcomm- or NVIDIA-certified
model. On the frozen encoded-video panel, the recommended 2 FPS profile scored
7/8 with a 1.453-second mean warm request time. The task-specific warehouse
classifier scored 4/4 at about 2.90 seconds per clip. See the
[engineering appendix](engineering_appendix.md) for the failed approaches,
precision and placement ablations, full benchmark history, and unresolved
runtime limits.

## What Cosmos-Reason2-2B is

[`Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B) is
NVIDIA's compact vision-language reasoning model for physical AI. It accepts a
prompt together with an image or video and generates text grounded in the
scene. Typical tasks include describing motion, locating an event in time,
recognizing a safety hazard, explaining an action, and predicting what is
likely to happen next in robotics, autonomous-driving, or smart-space footage.
It does not generate video and it is not a safety-certified controller.

The model is an interesting edge target because its 2.44 billion parameters
are small enough to quantize into a roughly 2 GiB GGUF bundle while retaining
the image/video-to-text interface and physical-reasoning specialization.
Keeping inference near a robot can reduce network latency, keep camera data on
the device, and improve performance per watt. It is still a demanding mobile
workload: retaining useful temporal and visual accuracy requires careful
quantization, preprocessing, context sizing, and NPU placement.

The deployed architecture has a 24-layer, width-1024 vision transformer and a
28-layer, width-2048 language transformer. A 224 × 384 frame is divided into
16 × 16 patches and merged in 2 × 2 groups, producing 84 visual tokens. The
vision layers turn those patches into scene features; a projector maps them
to the language model's 2048-wide token space. Each language layer then uses
attention to relate prompt and visual tokens, followed by a 6144-wide
feed-forward transformation. Sixteen query heads capture several
relationships in parallel, while eight shared key/value heads reduce cache
size and memory traffic. Three DeepStack connections inject intermediate
vision features into later language processing instead of relying only on the
final vision output. GenieX/llama.cpp keeps that multimodal wiring internal,
which is one reason this route proved more practical than exposing those
tensors across separately compiled QAIRT graphs.

## Before you start

You need:

- An x86-64 Linux host, or Ubuntu under WSL2, with Git, Git LFS, Python 3,
  CMake, a C/C++ toolchain, Docker, Bazelisk, `curl`, and `unzip`.
- Enough host disk for the 4.6 GiB checkpoint, intermediate BF16 GGUF, build
  trees, and the approximately 2 GiB deployable model. A GPU is not required
  for this GGUF route.
- A Hugging Face account that has accepted NVIDIA's gated model terms.
- A recursive checkout of this repository.
- An IQ-9075 Ubuntu EVK reachable as `ubuntu@<EVK IP>`, with its Qualcomm
  FastRPC/Hexagon driver stack working.

All host commands below run in Linux or WSL. Choose new output directories if
you already have files at these paths.

```bash
export REPO_ROOT=/path/to/qai-physics-reasoning
export WORK_ROOT="$HOME/cosmos-reason2-iq9"
export COSMOS_OFFICIAL="$WORK_ROOT/models/Cosmos-Reason2-2B"
export COSMOS_GGUF_BUILD="$WORK_ROOT/models/cosmos-reason2-2b-gguf-build"
export MODEL_DIR="$WORK_ROOT/models/cosmos-reason2-2b-geniex-q4_0"

mkdir -p "$WORK_ROOT/models"
cd "$REPO_ROOT"
```

Never put a Hugging Face token, Qualcomm token, SSH private key, or EVK IP
address in a tracked file.

## 1. Download and validate the official checkpoint

Create a small conversion environment, authenticate interactively, and
download the gated checkpoint:

```bash
python3 -m venv "$WORK_ROOT/venv"
source "$WORK_ROOT/venv/bin/activate"
python -m pip install --upgrade pip "huggingface_hub[cli]"

hf auth login
hf auth whoami
hf download nvidia/Cosmos-Reason2-2B \
  --local-dir "$COSMOS_OFFICIAL"

python "$REPO_ROOT/scripts/validate_model_config.py" "$COSMOS_OFFICIAL"
```

The validator must finish with:

```text
Compatible Qwen3-VL-2B checkpoint: .../Cosmos-Reason2-2B/config.json
```

Also confirm the essential files:

```bash
test -f "$COSMOS_OFFICIAL/config.json"
test -f "$COSMOS_OFFICIAL/model.safetensors"
test -f "$COSMOS_OFFICIAL/tokenizer.json"
test -f "$COSMOS_OFFICIAL/preprocessor_config.json"
```

Do not use a third-party GGUF as the conversion source and do not edit
`config.json` to bypass the validator.

## 2. Export the Q4_0 model and F16 vision projector

Use the converter revision that produced the verified artifacts:

```bash
export LLAMA_CPP="$WORK_ROOT/src/llama.cpp-910196f6"

git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP"
git -C "$LLAMA_CPP" checkout \
  910196f6b3dfc6aca88fa732e2b02f270ff9b56b

python -m pip install -r "$LLAMA_CPP/requirements.txt"

cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF
cmake --build "$LLAMA_CPP/build" \
  --config Release \
  --target llama-quantize llama-mtmd-cli \
  -j "$(nproc)"

mkdir -p "$COSMOS_GGUF_BUILD" "$MODEL_DIR"
```

Export the main model to BF16, export the multimodal projector to F16, then
quantize only the main model:

```bash
python "$LLAMA_CPP/convert_hf_to_gguf.py" \
  "$COSMOS_OFFICIAL" \
  --outfile "$COSMOS_GGUF_BUILD/Cosmos-Reason2-2B-BF16.gguf" \
  --outtype bf16

python "$LLAMA_CPP/convert_hf_to_gguf.py" \
  "$COSMOS_OFFICIAL" \
  --mmproj \
  --outfile "$MODEL_DIR/mmproj-Cosmos-Reason2-2B-F16.gguf" \
  --outtype f16

"$LLAMA_CPP/build/bin/llama-quantize" \
  "$COSMOS_GGUF_BUILD/Cosmos-Reason2-2B-BF16.gguf" \
  "$MODEL_DIR/Cosmos-Reason2-2B-Q4_0.gguf" \
  Q4_0
```

Verify the two deployable files:

```bash
sha256sum \
  "$MODEL_DIR/Cosmos-Reason2-2B-Q4_0.gguf" \
  "$MODEL_DIR/mmproj-Cosmos-Reason2-2B-F16.gguf"
```

Expected:

| File | Bytes | SHA-256 |
|---|---:|---|
| `Cosmos-Reason2-2B-Q4_0.gguf` | 1,229,455,008 | `90bd38f1a117b2d9738642165dfd8c0c6273a4bdfd214e1354d0502cff61e8a9` |
| `mmproj-Cosmos-Reason2-2B-F16.gguf` | 819,395,424 | `43df50be60fd6c9075548cd6d759e21253a3c62da69055bab7864cc64aa00c2c` |

Keep only these two files in `MODEL_DIR`. In particular, do not place the
intermediate BF16 main model there: GenieX local import expects one main GGUF
beside one `mmproj-*.gguf`. The standard Q4_0 recipe intentionally retains
the large output projection in Q6_K for quality; use it instead of the
experimental pure-Q4_0 variant.

## 3. Patch and build GenieX v0.3.17

Stock GenieX can load the model and images, but its encoded-video path leaks
the lazy video owner and does not always reap `ffmpeg`/`ffprobe`. It also
accepts server grammar fields without forwarding them to the sampler. Apply
the repository's four small fixes before building:

```bash
export GENIEX_SOURCE="$WORK_ROOT/src/GenieX-v0.3.17-native-video"

git clone --branch v0.3.17 --recurse-submodules \
  https://github.com/qualcomm/GenieX.git \
  "$GENIEX_SOURCE"
git -C "$GENIEX_SOURCE" checkout \
  2c2bde3afabc476f31e77bd7e01559c7e8f13cfd
git -C "$GENIEX_SOURCE" submodule update --init --recursive

for patch in \
  "$REPO_ROOT"/integrations/geniex_native_video/patches/*.patch
do
  git -C "$GENIEX_SOURCE" apply --check --unidiff-zero "$patch"
  git -C "$GENIEX_SOURCE" apply --unidiff-zero "$patch"
done
```

Cross-compile the ARM64 Snapdragon SDK with native video enabled. This is the
toolchain image used for the verified build:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --volume "$GENIEX_SOURCE:/workspace" \
  --workdir /workspace/sdk \
  --env CCACHE_DIR=/workspace/.ccache \
  --platform linux/amd64 \
  docker.io/qualcomm/geniex-toolchain-linux@sha256:75a7b808752b953aa454c0c4bbd3b8d6ab6ed29745ac423c2c00c92b3fd70b51 \
  bash -c '
    cmake --preset arm64-linux-snapdragon-release \
      -B build-native-video . \
      -DMTMD_VIDEO=ON &&
    cmake --build build-native-video -j "$(nproc)" &&
    cmake --install build-native-video --prefix pkg-geniex
  '
```

The patched grammar forwarding is in the Go CLI, so build the CLI after the
SDK has been installed at `sdk/pkg-geniex`:

```bash
cd "$GENIEX_SOURCE"
bazelisk build --config=linux_arm64 //cli:artifact

export GENIEX_ARTIFACT="$GENIEX_SOURCE/bazel-bin/cli/artifact.zip"
test -f "$GENIEX_ARTIFACT"
unzip -l "$GENIEX_ARTIFACT" | grep -E \
  'geniex$|libgeniex.so|llama_cpp/libggml-hexagon.so|llama_cpp/libmtmd.so'
```

The archive contains the ARM64 `geniex` executable, `libgeniex.so`, and the
`llama_cpp` and `qairt` plugin directories in one deployable layout.

## 4. Copy the runtime and model to the EVK

On the host:

```bash
export EVK_IP="<EVK IP>"
export EVK_USER=ubuntu
export EVK_SSH_KEY=/path/to/evk_ssh_key
export EVK_TARGET="${EVK_USER}@${EVK_IP}"

ssh -i "$EVK_SSH_KEY" "$EVK_TARGET" \
  'mkdir -p /home/ubuntu/deploy /home/ubuntu/models'

scp -i "$EVK_SSH_KEY" \
  "$GENIEX_ARTIFACT" \
  "${EVK_TARGET}:/home/ubuntu/deploy/geniex-native-video-v0.3.17.zip"

scp -i "$EVK_SSH_KEY" -r \
  "$MODEL_DIR" \
  "${EVK_TARGET}:/home/ubuntu/models/"
```

Connect to the EVK:

```bash
ssh -i "$EVK_SSH_KEY" "$EVK_TARGET"
```

Install the runtime dependencies and unpack into a new directory:

```bash
sudo apt update
sudo apt install -y \
  ffmpeg \
  libatomic1 \
  libglib2.0-0 \
  ocl-icd-libopencl1 \
  qcom-fastrpc1 \
  unzip

export GENIEX_ROOT=/home/ubuntu/deploy/geniex-native-video-v0.3.17
test ! -e "$GENIEX_ROOT"
mkdir "$GENIEX_ROOT"
unzip -q \
  /home/ubuntu/deploy/geniex-native-video-v0.3.17.zip \
  -d "$GENIEX_ROOT"
chmod +x "$GENIEX_ROOT/geniex"
```

If `GENIEX_ROOT` already exists, select a new versioned directory rather than
unpacking over a live runtime.

## 5. Import the local model

Still on the EVK:

```bash
export GENIEX_ROOT=/home/ubuntu/deploy/geniex-native-video-v0.3.17
export GENIEX_DATADIR=/home/ubuntu/geniex-cosmos-data
export MODEL_DIR=/home/ubuntu/models/cosmos-reason2-2b-geniex-q4_0
export LD_LIBRARY_PATH="$GENIEX_ROOT:$GENIEX_ROOT/llama_cpp"
export GENIEX_PLUGIN_PATH="$GENIEX_ROOT"

"$GENIEX_ROOT/geniex" --version
"$GENIEX_ROOT/geniex" \
  --data-dir "$GENIEX_DATADIR" \
  pull local/cosmos-reason2-2b:Q4_0 \
  --model-hub localfs \
  --local-path "$MODEL_DIR" \
  --model-type vlm
```

Import is needed only once for a given data directory. The model remains
subject to NVIDIA's license; do not add either GGUF to this repository.

## 6. Run an image smoke test

From a host shell, copy a PNG or JPEG to the EVK:

```bash
ssh -i "$EVK_SSH_KEY" "$EVK_TARGET" \
  'mkdir -p /home/ubuntu/test-data'
scp -i "$EVK_SSH_KEY" \
  /path/to/local/frame.png \
  "${EVK_TARGET}:/home/ubuntu/test-data/frame.png"
```

Then, in the EVK shell, use its absolute path in the prompt:

```bash
export TEST_IMAGE=/home/ubuntu/test-data/frame.png

"$GENIEX_ROOT/geniex" \
  --data-dir "$GENIEX_DATADIR" \
  --skip-update \
  infer local/cosmos-reason2-2b:Q4_0 \
  --compute npu \
  --ngl -1 \
  --nctx 4096 \
  --max-tokens 64 \
  --top-k 1 \
  --seed 42 \
  --think=false \
  --prompt "Describe the scene and identify the most immediate hazard. $TEST_IMAGE"
```

`--compute npu --ngl -1` requests NPU placement for the vision context and
all language-model layers. Use `--top-k 1 --seed 42` for deterministic
comparisons; GenieX v0.3.17 treats temperature zero as an unset value.

## 7. Start the persistent 2 FPS NPU video service

The service process should own the model for many independent requests.
Starting a new process per video repeatedly initializes and tears down the
Hexagon backend and proved unreliable.

```bash
export GENIEX_ROOT=/home/ubuntu/deploy/geniex-native-video-v0.3.17
export GENIEX_DATADIR=/home/ubuntu/geniex-cosmos-data
export LD_LIBRARY_PATH="$GENIEX_ROOT:$GENIEX_ROOT/llama_cpp"
export GENIEX_PLUGIN_PATH="$GENIEX_ROOT"
export MTMD_VIDEO_FPS=2
export MTMD_VIDEO_TIMESTAMP_INTERVAL_MS=0

"$GENIEX_ROOT/geniex" \
  --skip-update \
  serve \
  --host 127.0.0.1:18181 \
  --keepalive 3600 \
  --compute npu \
  --nctx 4096 \
  --ngl -1
```

Leave this terminal running. The server is deliberately loopback-only; send
requests from another EVK shell or use an SSH tunnel.

Two FPS is a measured model-quality setting, not just a performance setting.
One FPS lost box-pickup information, while three and four FPS reproduced
choice bias on the frozen panel. Disabling generic timestamp text lets
successive frames remain adjacent for Qwen3-VL's temporal-patch merger.

## 8. Send a native MP4 request

The server must be able to read the absolute video path. Copy the clip from
the host:

```bash
scp -i "$EVK_SSH_KEY" \
  /path/to/local/warehouse_clip.mp4 \
  "${EVK_TARGET}:/home/ubuntu/test-data/warehouse_clip.mp4"
```

Then, in a second EVK shell:

```bash
export VIDEO_URL=file:///home/ubuntu/test-data/warehouse_clip.mp4

curl --fail-with-body \
  --header 'Content-Type: application/json' \
  --header 'Connection: close' \
  --data @- \
  http://127.0.0.1:18181/v1/chat/completions <<JSON
{
  "model": "local/cosmos-reason2-2b:Q4_0",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "$VIDEO_URL"}},
      {"type": "text", "text": "Which event is directly visible? Answer one letter only: A) a forklift knocks down a striped safety marker; B) a worker picks up a box; C) a worker moves away from a forklift; D) multiple workers leave their aisles."}
    ]
  }],
  "max_completion_tokens": 4,
  "enable_think": false,
  "top_k": 1,
  "temperature": 0,
  "seed": 42,
  "grammar_string": "root ::= [ABCD]"
}
JSON
```

The grammar is appropriate only for a declared closed-choice task. It prevents
malformed output but cannot turn a wrong in-set prediction into a correct one.

For the four warehouse event families used by this project, copy the tracked
client from the host:

```bash
scp -i "$EVK_SSH_KEY" \
  "$REPO_ROOT/scripts/classify_geniex_warehouse_video.py" \
  "${EVK_TARGET}:/home/ubuntu/deploy/"
```

Then run it on the EVK:

```bash
python3 /home/ubuntu/deploy/classify_geniex_warehouse_video.py \
  --video /home/ubuntu/test-data/warehouse_clip.mp4
```

It makes two constrained requests and reports one of:

- `forklift_marker_knockdown`
- `routine_box_pickup`
- `forklift_human_near_miss`
- `multi_worker_aisle_exit`

This client is deliberately task-specific. Its 4/4 result does not imply
broad zero-shot video accuracy; a balanced direct four-choice diagnostic
scored 10/16.

## 9. Verify that the NPU backend is active

Do not treat the command-line flag alone as proof. While the service is
running, inspect the process on the EVK:

```bash
pid="$(pgrep -n -f 'geniex.*serve.*18181')"
test -n "$pid"

grep -E 'libggml-hexagon\.so|libcdsprpc\.so' "/proc/$pid/maps"
ls -l "/proc/$pid/fd" | grep '/dev/fastrpc-cdsp-secure'
```

The verified process maps the Hexagon GGML backend and Qualcomm FastRPC
library and holds the secure CDSP device. This proves live Hexagon use at the
runtime-placement boundary. It is not an operator-by-operator trace proving
that no unsupported operation ever falls back to the CPU.

## Expected result and operating limits

The recommended full-NPU encoded-video profile is:

| Setting | Value |
|---|---|
| Model | Standard `Cosmos-Reason2-2B-Q4_0.gguf` |
| Projector | `mmproj-Cosmos-Reason2-2B-F16.gguf` |
| Runtime | Patched GenieX v0.3.17 `llama_cpp` |
| Placement | `--compute npu --ngl -1` |
| Context | 4096 |
| Sampling | 2 FPS, timestamp interval 0 |
| Frozen panel | 7/8, answers `A C B A C B C A` |
| Warm request time | 1.453 s mean |
| Warehouse classifier | 4/4, about 2.90 s per clip |

Keep these limits in production experiments:

- Use one supervised persistent worker and independent HTTP connections.
- Recycle the worker conservatively before 40 video requests. A 48-request
  soak was stable, but a longer run stalled after 54 completed requests.
- If `/dev/fastrpc-cdsp-secure` disappears, stop retrying model creation and
  reboot or power-cycle the EVK.
- Keep the verified 384 × 216 input grid and `nctx=4096`. Larger visual grids
  and an 8192-token context triggered Hexagon failures in this runtime.
- Score every application on frozen, order-balanced clips. The model can
  produce plausible but incorrect physical predictions.

For the complete investigation—including the split QAIRT export, DeepStack
and `visual_pos_masks` issue, vision-bias repair, precision sweeps, CPU/NPU
placement study, native-video leak fixes, benchmark evolution, and failure
guide—continue with the
[engineering appendix](engineering_appendix.md).
