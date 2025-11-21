#!/usr/bin/env bash
set -euo pipefail

# Usage: ./pngs_to_gif.sh /path/to/frames output.gif
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 INPUT_FOLDER OUTPUT_GIF"
    exit 1
fi

INPUT_FOLDER="${1%/}"
OUTPUT_GIF="$2"
FPS=10  # Adjust frame rate here if needed

# Step 1: Generate palette
ffmpeg -framerate "$FPS" -i "$INPUT_FOLDER/frame_%03d.png" \
  -vf "fps=${FPS},palettegen=max_colors=256:stats_mode=full" \
  palette.png

# Step 2: Create GIF using palette
ffmpeg -thread_queue_size 64 -framerate "$FPS" -i "$INPUT_FOLDER/frame_%03d.png" -i palette.png \
  -filter_complex "fps=${FPS},paletteuse=dither=floyd_steinberg" \
  "$OUTPUT_GIF"

# Optional: clean up palette
rm -f palette.png

echo "GIF created successfully: $OUTPUT_GIF"
