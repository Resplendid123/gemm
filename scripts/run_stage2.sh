#!/bin/bash

set -euo pipefail

BENCH="./build/benchmark"
CONFIGS="./configs/stage2.txt"
RUNS=5

OUTPUT_CSV="./results/stage2_results.csv"
mkdir -p "$(dirname "$OUTPUT_CSV")"

BLOCK_CONFIGS=(
    "8 8"
    "16 16"
    "32 8"
    "8 32"
    "32 32"
)

echo "config_name,M,N,K,block_x,block_y,kernel,avg_time_ms,avg_gflops,rel_to_cublas_pct" > "$OUTPUT_CSV"

run_benchmark() {
    local M=$1
    local N=$2
    local K=$3
    local kernel=$4
    local block_x=${5:-}
    local block_y=${6:-}

    local total_time=0
    local total_gflops=0

    for _ in $(seq 1 "$RUNS"); do
        local out
        if [ -n "$block_x" ] && [ -n "$block_y" ]; then
            out=$($BENCH "$M" "$N" "$K" "$kernel" "$block_x" "$block_y" 2>&1)
        else
            out=$($BENCH "$M" "$N" "$K" "$kernel" 2>&1)
        fi

        local time=$(echo "$out" | grep -oP '[0-9.]+(?= ms)' | head -1)
        local gflops=$(echo "$out" | grep -oP 'GFLOPS: \K[0-9.]+' | head -1)

        total_time=$(echo "$total_time + ${time:-0}" | bc)
        total_gflops=$(echo "$total_gflops + ${gflops:-0}" | bc)
    done

    avg_time=$(echo "scale=6; $total_time / $RUNS" | bc)
    avg_gflops=$(echo "scale=6; $total_gflops / $RUNS" | bc)
    printf "%.3f\n%.2f" "$avg_time" "$avg_gflops"
}

while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue

    IFS='|' read -r cfg_name M N K <<< "$line"
    echo "Running config: $cfg_name ($M x $N x $K)"

    # cuBLAS 基准
    echo "  Testing cublas..."
    cublas_out=$(run_benchmark "$M" "$N" "$K" "cublas" 16 16)
    cublas_time=$(echo "$cublas_out" | sed -n '1p')
    cublas_gflops=$(echo "$cublas_out" | sed -n '2p')
    echo "$cfg_name,$M,$N,$K,NA,NA,cublas,$cublas_time,$cublas_gflops,100.00%" >> "$OUTPUT_CSV"

    for block_cfg in "${BLOCK_CONFIGS[@]}"; do
        read -r block_x block_y <<< "$block_cfg"
        echo "  Testing shared with block(${block_x},${block_y})..."

        shared_out=$(run_benchmark "$M" "$N" "$K" "shared" "$block_x" "$block_y")
        shared_time=$(echo "$shared_out" | sed -n '1p')
        shared_gflops=$(echo "$shared_out" | sed -n '2p')

        if [ -n "$cublas_gflops" ] && [ "$cublas_gflops" != "0" ]; then
            rel=$(echo "scale=2; $shared_gflops * 100 / $cublas_gflops" | bc)
        else
            rel="0.00"
        fi
        [[ $rel == .* ]] && rel="0$rel"

        echo "$cfg_name,$M,$N,$K,$block_x,$block_y,shared,$shared_time,$shared_gflops,$rel%" >> "$OUTPUT_CSV"
    done

done < "$CONFIGS"

echo -e "\n结果已保存到: $OUTPUT_CSV"
cat "$OUTPUT_CSV"
