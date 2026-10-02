#!/bin/bash
# Usage: bash run_all.sh [data_dir] > inference_logs_mem128_threads16.txt 2>&1
# Builds and evaluates HCL indexes for all datasets sequentially.
# Run from Hierarchical-Cut-Labelling/:
#
# Peak memory is measured via /usr/bin/time -v (Linux only), isolated per
# dataset process. Look for "Maximum resident set size" in the output.

set -e

DATA_DIR="${1:-../../../data}"
BUILD_DIR="./build"
SAVED_INDEXES_DIR="./saved_indexes"

DATASETS=(
    W_Jinan
    W_Shenzhen
    W_Chengdu
    W_Beijing
    W_Shanghai
    W_NewYork
    W_Chicago
    FLA
    E
    W
    CTR
    USA
)

mkdir -p "$SAVED_INDEXES_DIR"

for ds in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Dataset: $ds"
    echo "========================================"
    echo "Building HCL index for $ds..."
    /usr/bin/time -v \
        "$BUILD_DIR/index" "$DATA_DIR/$ds/$ds.edges" "$SAVED_INDEXES_DIR/$ds.hl" \
        2>&1
    echo "----------------------------------------"
    echo "Evaluating HCL index for $ds (1M random queries)..."
    /usr/bin/time -v \
        "$BUILD_DIR/query" "$SAVED_INDEXES_DIR/$ds.hl" \
        2>&1
done
