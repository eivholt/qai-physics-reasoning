#!/usr/bin/env bash
set -u

service_url="${EVK_GENIEX_URL:-http://127.0.0.1:18181}"
service_log="${EVK_GENIEX_LOG:-/home/ubuntu/deploy/geniex-isaac-serve-latest.log}"

printf 'EVK Cosmos / HTP monitor  %s\n' "$(date '+%F %T')"
printf '%s\n' '────────────────────────────────────────────────────────'

if models="$(curl --silent --show-error --fail --max-time 1 \
    "${service_url}/v1/models" 2>/dev/null)"; then
    model="$(
        printf '%s' "$models" |
            sed -n 's/.*"id":"\([^"]*\)".*/\1/p' |
            head -1
    )"
    printf 'GenieX endpoint : READY  %s\n' "${model:-model-id-unparsed}"
else
    printf 'GenieX endpoint : DOWN   %s\n' "$service_url"
fi

mapfile -t geniex_pids < <(
    ps -eo pid=,args= |
        awk '/geniex-cosmos-v0317-native-video-r1/ && !/awk/ {print $1}'
)
printf 'GenieX workers  : %d\n' "${#geniex_pids[@]}"

hexagon_mapped=0
fastrpc_open=0
for pid in "${geniex_pids[@]}"; do
    if grep -q 'libggml-hexagon\.so' "/proc/$pid/maps" 2>/dev/null; then
        hexagon_mapped=$((hexagon_mapped + 1))
    fi
    if ls -l "/proc/$pid/fd" 2>/dev/null |
        grep -q '/dev/fastrpc-cdsp-secure'; then
        fastrpc_open=$((fastrpc_open + 1))
    fi
done
printf 'Hexagon backend : mapped in %d process(es)\n' "$hexagon_mapped"
printf 'CDSP FastRPC    : open in %d process(es)\n' "$fastrpc_open"
printf 'HTP utilization : not exported as a %% counter by this kernel image\n'

printf '\nGenieX worker CPU (host-side submission/vision work):\n'
ps -eo pid=,pcpu=,pmem=,stat=,comm=,args= |
    awk '/geniex-cosmos-v0317-native-video-r1/ && !/awk/ {
        printf "  pid=%-6s cpu=%5s%% mem=%4s%% state=%-4s %s\n",
            $1, $2, $3, $4, $5
    }' |
    head -12

printf '\nNSP thermal sensors (NPU-adjacent, °C):\n  '
first=1
for hwmon in /sys/class/hwmon/hwmon*; do
    name="$(cat "$hwmon/name" 2>/dev/null || true)"
    case "$name" in
        nsp_*)
            raw="$(cat "$hwmon/temp1_input" 2>/dev/null || true)"
            test -n "$raw" || continue
            if test "$first" -eq 0; then
                printf '  '
            fi
            first=0
            awk -v name="$name" -v raw="$raw" \
                'BEGIN { printf "%s=%.1f", name, raw / 1000 }'
            ;;
    esac
done
printf '\n'

printf '\nLatest inference requests (endpoint busy time):\n'
if test -r "$service_log"; then
    grep 'POST.*"/v1/chat/completions"' "$service_log" |
        tail -6 |
        sed 's/^/  /'
else
    printf '  Service log is unavailable: %s\n' "$service_log"
fi

printf '\nInterpretation: endpoint busy time is pipeline duty, not HTP occupancy.\n'
