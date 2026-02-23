#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: mp4_batch10_to_jpg.sh <input_dir> [options]

Convert MP4 videos to JPG image sequences in batches.
Every batch of N videos shares one basename and one continuous frame counter.

Output sizing:
- Final frame size is always 1280x720.
- 16:9 sources are scaled directly to 1280x720.
- Non-16:9 sources are scaled proportionally, then padded to 1280x720.
- Padding is edge-clamped (smear), not flat black.

Default behavior (N=10):
- videos 1..10   -> batch_001/batch_001.000001.jpg ...
- videos 11..20  -> batch_002/batch_002.000001.jpg ...
- etc.

Options:
  --output-dir DIR      Output root directory (default: ./sequences)
  --batch-size N        Videos per batch (default: 10)
  --fps N               Extract at fixed fps (default: source fps)
  --quality N           JPEG quality q:v (1=best, 31=worst, default: 2)
  --prefix NAME         Batch basename prefix (default: batch)
  --edge-crop-lr PX     Crop PX from left and right before scaling (default: 5)
  -h, --help            Show this help

Example:
  ./mp4_batch10_to_jpg.sh "/home/adam/Downloads/TO ADAM" --batch-size 10 --quality 2 --prefix to_adam_batch
EOF
  exit 1
}

[[ $# -lt 1 ]] && usage

INPUT_DIR=""
OUTPUT_DIR="$(pwd)/sequences"
BATCH_SIZE=10
FPS=""
QUALITY=2
PREFIX="batch"
EDGE_CROP_LR=5
TARGET_W=1280
TARGET_H=720

ensure_even() {
  local n=$1
  if (( n % 2 == 0 )); then
    echo "$n"
  else
    echo $((n + 1))
  fi
}

ensure_even_floor() {
  local n=$1
  if (( n % 2 == 0 )); then
    echo "$n"
  else
    echo $((n - 1))
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --quality)
      QUALITY="$2"
      shift 2
      ;;
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --edge-crop-lr)
      EDGE_CROP_LR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      if [[ -z "$INPUT_DIR" ]]; then
        INPUT_DIR="$1"
        shift
      else
        echo "Unexpected argument: $1" >&2
        usage
      fi
      ;;
  esac
done

[[ -z "$INPUT_DIR" ]] && usage
[[ ! -d "$INPUT_DIR" ]] && { echo "Input directory not found: $INPUT_DIR" >&2; exit 1; }
[[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || { echo "--batch-size must be an integer" >&2; exit 1; }
(( BATCH_SIZE > 0 )) || { echo "--batch-size must be > 0" >&2; exit 1; }
[[ "$QUALITY" =~ ^[0-9]+$ ]] || { echo "--quality must be an integer" >&2; exit 1; }
(( QUALITY >= 1 && QUALITY <= 31 )) || { echo "--quality must be between 1 and 31" >&2; exit 1; }
[[ "$EDGE_CROP_LR" =~ ^[0-9]+$ ]] || { echo "--edge-crop-lr must be an integer" >&2; exit 1; }

command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg not found in PATH" >&2; exit 1; }
command -v ffprobe >/dev/null 2>&1 || { echo "ffprobe not found in PATH" >&2; exit 1; }

mapfile -d '' VIDEOS < <(find "$INPUT_DIR" -maxdepth 1 -type f -iname '*.mp4' -print0 | sort -z)
TOTAL_VIDEOS=${#VIDEOS[@]}

if (( TOTAL_VIDEOS == 0 )); then
  echo "No .mp4 files found in: $INPUT_DIR"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
MANIFEST="$OUTPUT_DIR/${PREFIX}_manifest.csv"

echo "video_path,batch_name,start_frame,end_frame,frames_added,src_w,src_h,crop_left,crop_right,work_w,work_h,scaled_w,scaled_h,pad_left,pad_right,pad_top,pad_bottom" > "$MANIFEST"

declare -A BATCH_COUNTER

echo "Found $TOTAL_VIDEOS videos"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "JPEG quality: $QUALITY"
echo "Target size: ${TARGET_W}x${TARGET_H}"
echo "Edge crop (L/R): ${EDGE_CROP_LR}px"
[[ -n "$FPS" ]] && echo "FPS override: $FPS"

echo

for i in "${!VIDEOS[@]}"; do
  video="${VIDEOS[$i]}"
  batch_num=$(( i / BATCH_SIZE + 1 ))
  batch_name=$(printf "%s_%03d" "$PREFIX" "$batch_num")
  batch_dir="$OUTPUT_DIR/$batch_name"
  mkdir -p "$batch_dir"

  current_count=${BATCH_COUNTER[$batch_name]:-0}
  start_frame=$(( current_count + 1 ))

  dims=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "$video" | head -n 1)
  src_w=${dims%x*}
  src_h=${dims#*x}

  if [[ -z "$src_w" || -z "$src_h" || "$src_w" == "$dims" ]]; then
    echo "[ERROR] Could not read dimensions for: $video" >&2
    exit 1
  fi

  crop_left=0
  crop_right=0
  if (( src_w > EDGE_CROP_LR * 2 )); then
    crop_left=$EDGE_CROP_LR
    crop_right=$EDGE_CROP_LR
  fi

  work_w=$(( src_w - crop_left - crop_right ))
  work_h=$src_h

  scaled_w=$TARGET_W
  scaled_h=$TARGET_H
  pad_left=0
  pad_right=0
  pad_top=0
  pad_bottom=0

  lhs=$(( work_w * TARGET_H ))
  rhs=$(( work_h * TARGET_W ))

  if (( lhs < rhs )); then
    # narrower than 16:9 -> keep height, pad left/right
    scaled_w=$(( (work_w * TARGET_H + work_h / 2) / work_h ))
    scaled_w=$(ensure_even "$scaled_w")
    (( scaled_w > TARGET_W )) && scaled_w=$TARGET_W

    total_pad_x=$(( TARGET_W - scaled_w ))
    pad_left=$(( total_pad_x / 2 ))
    pad_left=$(ensure_even_floor "$pad_left")
    pad_right=$(( total_pad_x - pad_left ))
  elif (( lhs > rhs )); then
    # wider than 16:9 -> keep width, pad top/bottom
    scaled_h=$(( (work_h * TARGET_W + work_w / 2) / work_w ))
    scaled_h=$(ensure_even "$scaled_h")
    (( scaled_h > TARGET_H )) && scaled_h=$TARGET_H

    total_pad_y=$(( TARGET_H - scaled_h ))
    pad_top=$(( total_pad_y / 2 ))
    pad_top=$(ensure_even_floor "$pad_top")
    pad_bottom=$(( total_pad_y - pad_top ))
  fi

  vf_parts=()
  if (( crop_left > 0 || crop_right > 0 )); then
    vf_parts+=("crop=${work_w}:${work_h}:${crop_left}:0")
  fi
  [[ -n "$FPS" ]] && vf_parts+=("fps=${FPS}")
  vf_parts+=("scale=${scaled_w}:${scaled_h}")

  if (( pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 )); then
    vf_parts+=("pad=${TARGET_W}:${TARGET_H}:${pad_left}:${pad_top}:black")
    vf_parts+=("fillborders=left=${pad_left}:right=${pad_right}:top=${pad_top}:bottom=${pad_bottom}:mode=smear")
  fi

  vf_filter=$(IFS=, ; echo "${vf_parts[*]}")

  before_count=$(find "$batch_dir" -maxdepth 1 -type f -name "$batch_name.*.jpg" | wc -l)

  echo "[$((i + 1))/$TOTAL_VIDEOS] $(basename "$video") -> $batch_name (start frame $start_frame)"
  echo "  src=${src_w}x${src_h} crop L${crop_left} R${crop_right} -> work=${work_w}x${work_h} scaled=${scaled_w}x${scaled_h} pad L${pad_left} R${pad_right} T${pad_top} B${pad_bottom}"

  ffmpeg -hide_banner -loglevel warning -stats -y \
    -i "$video" \
    -vf "$vf_filter" \
    -q:v "$QUALITY" \
    -start_number "$start_frame" \
    "$batch_dir/$batch_name.%06d.jpg"

  after_count=$(find "$batch_dir" -maxdepth 1 -type f -name "$batch_name.*.jpg" | wc -l)
  added=$(( after_count - before_count ))

  if (( added < 0 )); then
    echo "[ERROR] Frame count went backwards for $batch_name" >&2
    exit 1
  fi

  end_frame=$(( start_frame + added - 1 ))
  BATCH_COUNTER[$batch_name]=$(( current_count + added ))

  safe_video=${video//\"/\"\"}
  echo "\"$safe_video\",$batch_name,$start_frame,$end_frame,$added,$src_w,$src_h,$crop_left,$crop_right,$work_w,$work_h,$scaled_w,$scaled_h,$pad_left,$pad_right,$pad_top,$pad_bottom" >> "$MANIFEST"
done

echo
echo "Done."
echo "Manifest: $MANIFEST"
for batch in "${!BATCH_COUNTER[@]}"; do
  echo "  $batch: ${BATCH_COUNTER[$batch]} frames"
done
