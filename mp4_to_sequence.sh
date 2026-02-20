#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $(basename "$0") <video.mp4> [--fps N]"
    echo
    echo "Convert an MP4 video to a PNG image sequence."
    echo "Output: sequences/<moviename>/<moviename>.0001.png, .0002.png, ..."
    echo
    echo "Options:"
    echo "  --fps N   Extract at N frames per second (default: original fps)"
    echo
    echo "Examples:"
    echo "  $(basename "$0") clip.mp4"
    echo "  $(basename "$0") /path/to/render.mp4 --fps 24"
    exit 1
}

[[ $# -lt 1 ]] && usage

VIDEO=""
FPS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fps)
            FPS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            VIDEO="$1"
            shift
            ;;
    esac
done

[[ -z "$VIDEO" ]] && usage
[[ ! -f "$VIDEO" ]] && echo "Error: File not found: $VIDEO" && exit 1

# Get the movie name (filename without extension)
MOVIENAME="$(basename "${VIDEO%.*}")"

# Output directory next to this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="${SCRIPT_DIR}/sequences/${MOVIENAME}"

mkdir -p "$OUTDIR"

# Build ffmpeg filter
FPS_FILTER=""
if [[ -n "$FPS" ]]; then
    FPS_FILTER="-vf fps=${FPS}"
fi

echo "Converting: $VIDEO"
echo "Output:     $OUTDIR/${MOVIENAME}.%04d.png"
[[ -n "$FPS" ]] && echo "FPS:        $FPS"

# shellcheck disable=SC2086
ffmpeg -i "$VIDEO" $FPS_FILTER "$OUTDIR/${MOVIENAME}.%04d.png" -y -loglevel warning -stats

COUNT=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
echo "Done! Extracted $COUNT frames to $OUTDIR/"
