#!/usr/bin/env bash
set -euo pipefail

GENIEX_ROOT="${GENIEX_ROOT:-/home/ubuntu/geniex-cosmos-v0317-native-video-r1}"
GENIEX_DATADIR="${GENIEX_DATADIR:-/home/ubuntu/geniex-cosmos-v0317-data}"
SERVICE_HOST="${SERVICE_HOST:-127.0.0.1:18181}"
LOG_DIR="${LOG_DIR:-/home/ubuntu/deploy}"
VIDEO_FPS="${MTMD_VIDEO_FPS:-2}"
VIDEO_TIMESTAMP_INTERVAL_MS="${MTMD_VIDEO_TIMESTAMP_INTERVAL_MS:-0}"
MODEL_ID="local/cosmos-reason2-2b"

if curl --silent --show-error --fail --max-time 3 \
    "http://${SERVICE_HOST}/v1/models" >/dev/null 2>&1; then
    printf 'GenieX is already ready at http://%s\n' "$SERVICE_HOST"
    exit 0
fi

if pgrep -f "${GENIEX_ROOT}/geniex-grammar.*serve.*${SERVICE_HOST}" \
    >/dev/null 2>&1; then
    printf 'A GenieX process already owns the requested service command but is not ready.\n' >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_path="${LOG_DIR}/geniex-isaac-serve-${timestamp}.log"
ln -sfn "$log_path" "${LOG_DIR}/geniex-isaac-serve-latest.log"

nohup env \
    GENIEX_DATADIR="$GENIEX_DATADIR" \
    LD_LIBRARY_PATH="${GENIEX_ROOT}:${GENIEX_ROOT}/llama_cpp" \
    GENIEX_PLUGIN_PATH="$GENIEX_ROOT" \
    MTMD_VIDEO_FPS="$VIDEO_FPS" \
    MTMD_VIDEO_TIMESTAMP_INTERVAL_MS="$VIDEO_TIMESTAMP_INTERVAL_MS" \
    "${GENIEX_ROOT}/geniex-grammar" \
        --skip-update \
        serve \
        --host "$SERVICE_HOST" \
        --keepalive 3600 \
        --compute npu \
        --nctx 4096 \
        --ngl -1 \
    >"$log_path" 2>&1 </dev/null &
service_pid=$!
printf 'Started GenieX PID %s; log: %s\n' "$service_pid" "$log_path"

for _ in $(seq 1 90); do
    if ! kill -0 "$service_pid" 2>/dev/null; then
        printf 'GenieX exited before becoming ready.\n' >&2
        tail -80 "$log_path" >&2
        exit 1
    fi
    models="$(
        curl --silent --show-error --fail --max-time 3 \
            "http://${SERVICE_HOST}/v1/models" 2>/dev/null || true
    )"
    if [[ "$models" == *"${MODEL_ID}"* ]]; then
        printf 'GenieX is ready with %s\n' "$MODEL_ID"
        exit 0
    fi
    sleep 1
done

printf 'GenieX did not become ready within 90 seconds.\n' >&2
tail -80 "$log_path" >&2
exit 1
