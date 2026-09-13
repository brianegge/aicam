#!/bin/bash
# Stretch ipcams2 v32 to 608x608 to match camera.py's inference preprocessing.
#
# camera.py does cv2.resize(image, IPCAMS_INPUT_SIZE) -- an unconditional
# stretch, not a letterbox. Training on letterboxed images would teach the
# model an aspect ratio it never sees live. YOLO labels are normalised, so a
# pure stretch leaves them correct and they are copied verbatim.
set -euo pipefail
SRC=/mnt/storage/ipcams2/v32
DST=/mnt/storage/ipcams2/v32-608
for s in train valid test; do
  mkdir -p "$DST/$s/images" "$DST/$s/labels"
  cp -a "$SRC/$s/labels/." "$DST/$s/labels/"
  ls "$SRC/$s/images" | xargs -P 8 -I{} sh -c \
    'ffmpeg -v error -y -i "'"$SRC/$s/images"'/{}" -vf scale=608:608 -q:v 2 "'"$DST/$s/images"'/{}"'
  echo "$s done: $(ls "$DST/$s/images" | wc -l) images"
done
cat > "$DST/data.yaml" <<YAML
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 8
names: ['cat', 'coyote', 'deer', 'dog', 'fox', 'person', 'rabbit', 'raccoon']

# Built by build_608.sh from ipcams2 v32 (native resolution export).
# Every image stretched to 608x608 to match camera.py resize().
YAML
echo "=== build complete ==="
