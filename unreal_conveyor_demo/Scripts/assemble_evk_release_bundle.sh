#!/usr/bin/env bash
# Assemble the private, self-contained EVK installer payload from verified local artifacts.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 /path/to/geniex-artifact.zip /path/to/model-dir /path/to/output.tar.gz" >&2
    exit 2
fi

artifact_zip="$(realpath "$1")"
model_dir="$(realpath "$2")"
output="$(realpath -m "$3")"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

main_model="Cosmos-Reason2-2B-Q4_0.gguf"
projector="mmproj-Cosmos-Reason2-2B-F16.gguf"

[[ -f "$artifact_zip" ]] || { echo "Missing GenieX artifact: $artifact_zip" >&2; exit 1; }
[[ -f "$model_dir/$main_model" ]] || { echo "Missing EVK model: $model_dir/$main_model" >&2; exit 1; }
[[ -f "$model_dir/$projector" ]] || { echo "Missing EVK projector: $model_dir/$projector" >&2; exit 1; }

mkdir -p "$work_dir/payload/geniex" "$work_dir/payload/models" "$(dirname "$output")"
unzip -q "$artifact_zip" -d "$work_dir/payload/geniex"
mv "$work_dir/payload/geniex/geniex" "$work_dir/payload/geniex/geniex-grammar"
chmod 0755 "$work_dir/payload/geniex/geniex-grammar"
cp "$model_dir/$main_model" "$work_dir/payload/models/$main_model"
cp "$model_dir/$projector" "$work_dir/payload/models/$projector"

(
    cd "$work_dir/payload"
    {
        echo "format=qai-conveyor-evk-bundle-v1"
        echo "geniex_commit=2c2bde3afabc476f31e77bd7e01559c7e8f13cfd"
        echo "llama_cpp_commit=ae9291e16b976514bf1f3d7f1616da7db3459496"
        sha256sum "models/$main_model" "models/$projector"
    } >bundle_manifest.txt
    tar --sort=name --mtime='UTC 2026-01-01' --owner=0 --group=0 --numeric-owner \
        -czf "$output.tmp" bundle_manifest.txt geniex models
)
mv "$output.tmp" "$output"

tar -tzf "$output" >"$work_dir/archive-entries.txt"
grep -Fxq 'geniex/geniex-grammar' "$work_dir/archive-entries.txt"
grep -Fxq "models/$main_model" "$work_dir/archive-entries.txt"
grep -Fxq "models/$projector" "$work_dir/archive-entries.txt"
sha256sum "$output"
du -h "$output"
