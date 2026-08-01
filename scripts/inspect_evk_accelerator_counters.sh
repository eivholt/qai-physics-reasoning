#!/usr/bin/env bash
set -u

printf '== installed tools ==\n'
for tool in htop btop atop pidstat perf trace-cmd qnn-profile-viewer qnn-net-run \
    snpe-diagview powertop; do
    command -v "$tool" 2>/dev/null || true
done

printf '\n== devfreq devices ==\n'
ls -1 /sys/class/devfreq 2>/dev/null || true

printf '\n== accelerator-related sysfs paths ==\n'
find /sys/class /sys/devices/platform -maxdepth 7 \
    \( -ipath '*npu*' -o -ipath '*cdsp*' -o -ipath '*htp*' \
       -o -ipath '*hexagon*' -o -ipath '*fastrpc*' \) \
    2>/dev/null | head -200

printf '\n== accelerator-related debugfs paths ==\n'
find /sys/kernel/debug -maxdepth 7 \
    \( -ipath '*npu*' -o -ipath '*cdsp*' -o -ipath '*htp*' \
       -o -ipath '*hexagon*' -o -ipath '*fastrpc*' \) \
    2>/dev/null | head -200

printf '\n== matching trace events ==\n'
grep -Ei '(fastrpc|cdsp|remoteproc|rpmsg|qnn|htp|hexagon)' \
    /sys/kernel/tracing/available_events 2>/dev/null | head -200 || true

printf '\n== matching interrupts ==\n'
grep -Ei '(fastrpc|cdsp|remoteproc|rpmsg|qnn|htp|hexagon)' \
    /proc/interrupts 2>/dev/null | head -100 || true

printf '\n== matching thermal zones ==\n'
for zone in /sys/class/thermal/thermal_zone*; do
    zone_type="$(cat "$zone/type" 2>/dev/null || true)"
    if printf '%s' "$zone_type" | grep -Eiq \
        '(npu|cdsp|dsp|htp|hexagon|q6)'; then
        printf '%s %s ' "$zone" "$zone_type"
        cat "$zone/temp" 2>/dev/null || true
    fi
done

printf '\n== hwmon labels ==\n'
for hwmon in /sys/class/hwmon/hwmon*; do
    printf '%s name=%s\n' "$hwmon" "$(cat "$hwmon/name" 2>/dev/null || true)"
    for label in "$hwmon"/*_label; do
        test -f "$label" || continue
        printf '  %s=%s\n' "$(basename "$label")" "$(cat "$label")"
    done
done
