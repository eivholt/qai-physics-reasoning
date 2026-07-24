#!/usr/bin/env bash
set -euo pipefail

# Build, package, deploy, or run the standalone full-DeepStack GenieX path.
# Nothing is built or copied unless the corresponding subcommand is invoked.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PINNED_GENIEX_COMMIT="ab9e904fe21a01673840495309f3471dd6980d10"
PINNED_QAIRT_COMMIT="f00a37cd50645456c8fbf0eaf179fa59640fd5bf"
PINNED_GENIEX_PROC_COMMIT="2ec6abcac6db8473fa26ea2f882f99652723cc7b"
DEFAULT_TOOLCHAIN_IMAGE="docker.io/qualcomm/geniex-toolchain-linux:v0.0.1@sha256:694ca209accd73859ae889a689f5c9df3f8f202074c55c57641f69150b794a8f"

usage() {
    cat <<'EOF'
Usage:
  scripts/geniex_raw_video.sh build \
    --geniex-source /path/to/GenieX --build-dir /new/build/dir [--jobs N]

  scripts/geniex_raw_video.sh package \
    --bundle /path/to/full-deepstack-bundle \
    --video-input-dir /path/to/prepare_video_npu_inputs/output \
    --package-dir /new/package/dir

  scripts/geniex_raw_video.sh deploy \
    --build-dir /path/to/build --package-dir /path/to/package \
    --evk ubuntu@<EVK-IP> --remote-root /home/ubuntu/new-run-directory

  scripts/geniex_raw_video.sh run \
    --evk ubuntu@<EVK-IP> --remote-root /home/ubuntu/new-run-directory \
    --bundle /home/ubuntu/full-deepstack-bundle [--max-tokens 64] [--verbose]

The build uses a writable copy under build-dir and applies the repository's
DeepStack tensor-classification patch there. It never edits the vendor
checkout. Deploy refuses to overwrite an existing remote root. The model
bundle is not copied by deploy; --bundle on run names an existing EVK bundle.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_value() {
    local flag="$1"
    local value="${2:-}"
    [[ -n "$value" ]] || die "missing value for $flag"
}

require_new_dir() {
    local path="$1"
    [[ ! -e "$path" ]] || die "refusing to overwrite existing path: $path"
    mkdir -p "$(dirname "$path")"
    mkdir "$path"
}

validate_remote() {
    local evk="$1"
    local path="$2"
    [[ "$evk" =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$ ]] ||
        die "--evk must look like ubuntu@<EVK-IP> or user@hostname"
    [[ "$path" =~ ^/home/[A-Za-z0-9._-]+/[A-Za-z0-9._/-]+$ ]] ||
        die "remote paths must be explicit descendants of /home/<user>"
    [[ "$path" != *"/../"* && "$path" != *"/.." ]] ||
        die "remote paths must not contain '..'"
}

command_build() {
    local geniex_source=""
    local build_dir=""
    local jobs=""
    local image="$DEFAULT_TOOLCHAIN_IMAGE"
    while (($#)); do
        case "$1" in
            --geniex-source)
                require_value "$1" "${2:-}"
                geniex_source="$2"
                shift 2
                ;;
            --build-dir)
                require_value "$1" "${2:-}"
                build_dir="$2"
                shift 2
                ;;
            --jobs)
                require_value "$1" "${2:-}"
                jobs="$2"
                shift 2
                ;;
            --toolchain-image)
                require_value "$1" "${2:-}"
                image="$2"
                shift 2
                ;;
            *)
                die "unknown build option: $1"
                ;;
        esac
    done
    [[ -n "$geniex_source" && -n "$build_dir" ]] ||
        die "build requires --geniex-source and --build-dir"
    geniex_source="$(cd "$geniex_source" && pwd)"
    build_dir="$(realpath -m "$build_dir")"
    [[ "$(git -C "$geniex_source" rev-parse HEAD)" == "$PINNED_GENIEX_COMMIT" ]] ||
        die "GenieX checkout is not pinned v0.3.16 commit $PINNED_GENIEX_COMMIT"
    [[ "$(git -C "$geniex_source/third-party/geniex-qairt" rev-parse HEAD)" == "$PINNED_QAIRT_COMMIT" ]] ||
        die "geniex-qairt submodule is not pinned commit $PINNED_QAIRT_COMMIT"
    [[ -f "$geniex_source/third-party/geniex-qairt/third-party/geniex-proc/CMakeLists.txt" ]] ||
        die "initialize the pinned checkout's recursive submodules first"
    [[ "$(git -C "$geniex_source/third-party/geniex-qairt/third-party/geniex-proc" rev-parse HEAD)" == "$PINNED_GENIEX_PROC_COMMIT" ]] ||
        die "geniex-proc submodule is not pinned commit $PINNED_GENIEX_PROC_COMMIT"
    for repository in \
        "$geniex_source" \
        "$geniex_source/third-party/geniex-qairt" \
        "$geniex_source/third-party/geniex-qairt/third-party/geniex-proc"; do
        [[ -z "$(git -C "$repository" status --porcelain=v1 --untracked-files=normal --ignore-submodules=none)" ]] ||
            die "build source is dirty: $repository"
    done
    local submodule_status
    submodule_status="$(
        git -C "$geniex_source/third-party/geniex-qairt" \
            submodule status --recursive
    )"
    if grep -Eq '^[+-U]' <<<"$submodule_status"; then
        die "geniex-qairt has missing or divergent recursive submodules"
    fi
    [[ -z "$jobs" || "$jobs" =~ ^[1-9][0-9]*$ ]] ||
        die "--jobs must be a positive integer"
    [[ "$image" =~ @sha256:[0-9a-f]{64}$ ]] ||
        die "--toolchain-image must be pinned by an immutable sha256 digest"
    require_new_dir "$build_dir"

    local job_arg='$(nproc)'
    if [[ -n "$jobs" ]]; then
        job_arg="$jobs"
    fi
    docker run --rm \
        --platform linux/amd64 \
        --user "$(id -u):$(id -g)" \
        --volume "$geniex_source:/geniex-readonly:ro" \
        --volume "$PROJECT_ROOT:/project:ro" \
        --volume "$build_dir:/out" \
        --workdir /out \
        "$image" \
        bash -lc "
            set -euo pipefail
            cp -a /geniex-readonly /out/geniex-patched
            cd /out/geniex-patched/third-party/geniex-qairt
            git apply --check \
                /project/integrations/geniex_raw_video/patches/0001-deepstack-inputs-are-special.patch
            git apply --whitespace=error \
                /project/integrations/geniex_raw_video/patches/0001-deepstack-inputs-are-special.patch
            grep -q 'deepstack_visual_embeds_' core/src/llm/llm_utils.cpp
            cmake \
                -S /project/integrations/geniex_raw_video \
                -B /out/cmake \
                -DGENIEX_SOURCE=/out/geniex-patched \
                -DCMAKE_TOOLCHAIN_FILE=/out/geniex-patched/sdk/cmake/arm64-linux-gnu.cmake \
                -DCMAKE_BUILD_TYPE=Release
            cmake --build /out/cmake \
                --target cosmos_geniex_raw_video \
                --parallel $job_arg
        "
    [[ -x "$build_dir/cmake/bin/cosmos_geniex_raw_video" ]] ||
        die "build completed without the expected runner"
    printf 'Built runner: %s\n' \
        "$build_dir/cmake/bin/cosmos_geniex_raw_video"
}

command_package() {
    local bundle=""
    local video_input_dir=""
    local package_dir=""
    while (($#)); do
        case "$1" in
            --bundle)
                require_value "$1" "${2:-}"
                bundle="$2"
                shift 2
                ;;
            --video-input-dir)
                require_value "$1" "${2:-}"
                video_input_dir="$2"
                shift 2
                ;;
            --package-dir)
                require_value "$1" "${2:-}"
                package_dir="$2"
                shift 2
                ;;
            *)
                die "unknown package option: $1"
                ;;
        esac
    done
    [[ -n "$bundle" && -n "$video_input_dir" && -n "$package_dir" ]] ||
        die "package requires --bundle, --video-input-dir, and --package-dir"
    python3 "$PROJECT_ROOT/scripts/prepare_geniex_raw_video_run.py" prepare \
        --bundle "$bundle" \
        --video-input-dir "$video_input_dir" \
        --output-dir "$package_dir"
}

command_deploy() {
    local build_dir=""
    local package_dir=""
    local evk=""
    local remote_root=""
    while (($#)); do
        case "$1" in
            --build-dir)
                require_value "$1" "${2:-}"
                build_dir="$2"
                shift 2
                ;;
            --package-dir)
                require_value "$1" "${2:-}"
                package_dir="$2"
                shift 2
                ;;
            --evk)
                require_value "$1" "${2:-}"
                evk="$2"
                shift 2
                ;;
            --remote-root)
                require_value "$1" "${2:-}"
                remote_root="$2"
                shift 2
                ;;
            *)
                die "unknown deploy option: $1"
                ;;
        esac
    done
    [[ -n "$build_dir" && -n "$package_dir" && -n "$evk" && -n "$remote_root" ]] ||
        die "deploy requires --build-dir, --package-dir, --evk, and --remote-root"
    validate_remote "$evk" "$remote_root"
    local runtime_dir="$build_dir/cmake/bin"
    [[ -x "$runtime_dir/cosmos_geniex_raw_video" ]] ||
        die "runner not found under $runtime_dir"
    [[ -f "$package_dir/geniex_raw_video_manifest.json" ]] ||
        die "run package has no manifest: $package_dir"

    if ssh "$evk" "test -e '$remote_root'"; then
        die "refusing to overwrite existing EVK path: $remote_root"
    fi
    ssh "$evk" "mkdir -p '$remote_root'"
    scp -r "$runtime_dir" "$evk:$remote_root/runtime"
    scp -r "$package_dir" "$evk:$remote_root/run-package"
    scp "$PROJECT_ROOT/scripts/prepare_geniex_raw_video_run.py" \
        "$evk:$remote_root/prepare_geniex_raw_video_run.py"
    printf 'Deployed new run directory: %s:%s\n' "$evk" "$remote_root"
}

command_run() {
    local evk=""
    local remote_root=""
    local bundle=""
    local max_tokens="64"
    local verbose=""
    while (($#)); do
        case "$1" in
            --evk)
                require_value "$1" "${2:-}"
                evk="$2"
                shift 2
                ;;
            --remote-root)
                require_value "$1" "${2:-}"
                remote_root="$2"
                shift 2
                ;;
            --bundle)
                require_value "$1" "${2:-}"
                bundle="$2"
                shift 2
                ;;
            --max-tokens)
                require_value "$1" "${2:-}"
                max_tokens="$2"
                shift 2
                ;;
            --verbose)
                verbose=" --verbose"
                shift
                ;;
            *)
                die "unknown run option: $1"
                ;;
        esac
    done
    [[ -n "$evk" && -n "$remote_root" && -n "$bundle" ]] ||
        die "run requires --evk, --remote-root, and --bundle"
    validate_remote "$evk" "$remote_root"
    validate_remote "$evk" "$bundle"
    [[ "$max_tokens" =~ ^[1-9][0-9]*$ ]] ||
        die "--max-tokens must be a positive integer"

    ssh "$evk" "
        cd '$remote_root/runtime'
        LD_LIBRARY_PATH='$remote_root/runtime:$remote_root/runtime/htp-files:/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu' \
        ADSP_LIBRARY_PATH='$remote_root/runtime/htp-files' \
        python3 '$remote_root/prepare_geniex_raw_video_run.py' launch \
            --package-dir '$remote_root/run-package' \
            --bundle '$bundle' \
            --runner '$remote_root/runtime/cosmos_geniex_raw_video' \
            --max-tokens '$max_tokens'$verbose
    "
}

if (($# == 0)); then
    usage
    exit 1
fi

subcommand="$1"
shift
case "$subcommand" in
    build) command_build "$@" ;;
    package) command_package "$@" ;;
    deploy) command_deploy "$@" ;;
    run) command_run "$@" ;;
    help | --help | -h) usage ;;
    *) die "unknown subcommand: $subcommand" ;;
esac
