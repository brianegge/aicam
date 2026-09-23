#!/bin/sh
# Trim saved capture frames.
#
# The nano ran a usage-threshold version of this against a dedicated /data
# partition. On claw-mini the capture directory shares the system volume with
# everything else, so a pure "delete when the disk hits 90%" rule would fire on
# growth that has nothing to do with aicam. Age out by day instead, and keep the
# usage check only as a safety valve.
#
# Capture directories are named YYYYMMDD by main.py's save-path.

set -eu

DIR="${AICAM_CAPTURE_DIR:-$HOME/aicam-data/capture}"
KEEP_DAYS="${AICAM_KEEP_DAYS:-14}"
FULL_PCT="${AICAM_FULL_PCT:-92}"

[ -d "$DIR" ] || exit 0

# Age-based pass: drop whole day directories older than KEEP_DAYS.
find "$DIR" -mindepth 1 -maxdepth 1 -type d -name '2[0-9][0-9][0-9][0-1][0-9][0-3][0-9]' \
    -mtime "+${KEEP_DAYS}" -print -exec rm -rf {} +

# review/ is flat, not named by day, and its own mtime is bumped every time a
# verdict frame lands in it, so the pass above never reaps it. Age its contents
# out file by file instead. Only review/ -- excludes-auto/ next to it holds live
# exclusion definitions that main.py loads at startup, and static/ is fixture
# data; neither ages out.
if [ -d "${DIR}/review" ]; then
    find "${DIR}/review" -type f -mtime "+${KEEP_DAYS}" -delete
fi

# Safety valve: while the volume is over FULL_PCT, drop the oldest day
# directory. Bounded so a disk filled by something else cannot spin here.
i=0
while [ "$i" -lt 60 ]; do
    usep=$(df -P "$DIR" | awk 'NR==2 { gsub(/%/, "", $5); print $5 }')
    [ "$usep" -ge "$FULL_PCT" ] || break
    oldest=$(ls -dtr "$DIR"/2[0-9][0-9][0-9][0-1][0-9][0-3][0-9] 2>/dev/null | head -n 1)
    [ -n "$oldest" ] && [ -d "$oldest" ] || break
    echo "$(date): ${DIR} at ${usep}%, removing ${oldest}"
    rm -rf "$oldest"
    i=$((i + 1))
done
