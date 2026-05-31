#!/bin/bash

# Stage 6: Tensor Core GEMM 实验脚本
set -euo pipefail

BENCH="./build/benchmark"
CONFIGS="./configs/stage.txt"
RUNS=5

OUTPUT_CSV="./results/stage6_results.csv"
mkdir -p "$(dirname "$OUTPUT_CSV")"

# Tensor Core 配置: config_id|bm|bn|precision
# BM=64, BN=64, BK=16, TM=16, TN=16 (固定)
TENSOR_CONFIGS=(
    "1|64|64|FP16"    # Config 1: FP16 输出
    "2|64|64|FP32"    # Config 2: FP32 输出
)

# 统一 CSV 格式: config_name,M,N,K,kernel,BM,BN,BK,TM,TN,avg_time_ms,avg_gflops,rel_to_cublas_pct
echo "config_name,M,N,K,kernel,BM,BN,BK,TM,TN,avg_time_ms,avg_gflops,rel_to_cublas_pct" > "$OUTPUT_CSV"

run_benchmark() {
    local M=$1
    local N=$2
    local K=$3
    local kernel=$4
    shift 4
    local extra_args="$@"

    local total_time=0
    local total_gflops=0

    for _ in $(seq 1 "$RUNS"); do
        local out=$($BENCH "$M" "$N" "$K" "$kernel" $extra_args 2>&1)
        local time=$(echo "$out" | grep -oP '[0-9.]+(?= ms)' | head -1)
        local gflops=$(echo "$out" | grep -oP 'GFLOPS: \K[0-9.]+' | head -1)

        total_time=$(echo "$total_time + ${time:-0}" | bc)
        total_gflops=$(echo "$total_gflops + ${gflops:-0}" | bc)
    done

    avg_time=$(echo "scale=6; $total_time / $RUNS" | bc)
    avg_gflops=$(echo "scale=2; $total_gflops / $RUNS" | bc)
    printf "%.3f\n%.2f" "$avg_time" "$avg_gflops"
}

if [ ! -f "$CONFIGS" ]; then
    echo "Error: Config file $CONFIGS not found!"
    exit 1
fi

while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    IFS='|' read -r cfg_name M N K <<< "$line"
    echo "=========================================="
    echo "Testing $cfg_name ($M x $N x $K)"
    echo "=========================================="

    # cuBLAS 基准
    echo "  [Reference] cuBLAS..."
    cublas_out=$(run_benchmark "$M" "$N" "$K" "cublas")
    cublas_time=$(echo "$cublas_out" | sed -n '1p')
    cublas_gflops=$(echo "$cublas_out" | sed -n '2p')
    echo "$cfg_name,$M,$N,$K,cublas,NA,NA,NA,NA,NA,$cublas_time,$cublas_gflops,100.00%" >> "$OUTPUT_CSV"
    echo "    cuBLAS: time=${cublas_time}ms, GFLOPS=${cublas_gflops}"

    # 测试所有 Tensor Core 配置
    for tensor_cfg in "${TENSOR_CONFIGS[@]}"; do
        IFS='|' read -r config_id bm bn precision <<< "$tensor_cfg"
        echo "  [Stage6] Tensor Core (config=$config_id: BM=$bm, BN=$bn, $precision)..."

        tensor_out=$(run_benchmark "$M" "$N" "$K" "tensor" "--config=$config_id")
        tensor_time=$(echo "$tensor_out" | sed -n '1p')
        tensor_gflops=$(echo "$tensor_out" | sed -n '2p')

        if [ -n "$cublas_gflops" ] && [ "$cublas_gflops" != "0" ]; then
            rel_tensor=$(echo "scale=2; $tensor_gflops * 100 / $cublas_gflops" | bc)
        else
            rel_tensor="0.00"
        fi
        [[ $rel_tensor == .* ]] && rel_tensor="0$rel_tensor"

        # Tensor Core: BM, BN, BK=16(WMMA_K), TM=16(WMMA_M), TN=16(WMMA_N)
        full_cfg_name="${cfg_name}_${precision}"
        echo "$full_cfg_name,$M,$N,$K,tensor,$bm,$bn,16,16,16,$tensor_time,$tensor_gflops,${rel_tensor}%" >> "$OUTPUT_CSV"
        echo "    Tensor Core ($precision): time=${tensor_time}ms, GFLOPS=${tensor_gflops}, rel=${rel_tensor}%"
    done

    echo ""

done < "$CONFIGS"

echo "=========================================="
echo "Results saved to: $OUTPUT_CSV"
echo "=========================================="
cat "$OUTPUT_CSV"
