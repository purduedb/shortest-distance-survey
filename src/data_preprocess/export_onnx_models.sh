#!/bin/bash
# Usage: bash export_onnx_models.sh <results_expt_dir>

EXPT_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JIT_DIR="$EXPT_DIR/saved_jit_models"
ONNX_DIR="$EXPT_DIR/saved_onnx_models"

for jit_file in "$JIT_DIR"/*.jit.pt; do
    name="$(basename "$jit_file" .jit.pt)"
    out_onnx="$ONNX_DIR/$name.onnx"

    if [ -f "$out_onnx" ]; then
        printf "[%s] Already exported, skipping\n" "$name"
        continue
    fi

    printf "[%s] Starting...\n" "$name"
    start_time=$SECONDS

    python "$SCRIPT_DIR/export_onnx_models.py" --model_path "$jit_file" --output_dir "$ONNX_DIR" --name "$name"

    elapsed_time=$(( SECONDS - start_time ))
    printf "[%s] Done in %02d:%02d:%02d\n\n" "$name" $((elapsed_time/3600)) $((elapsed_time%3600/60)) $((elapsed_time%60))
done
