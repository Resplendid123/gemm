#!/bin/bash

# Stage 0: GEMM baseline 测试
# 测试 cpu kernel 与 cuBLAS 的性能差异

set -euo pipefail

BENCH="./build/benchmark"
CONFIGS="./configs/stage.txt"
RUNS=5

OUTPUT_CSV="./results/stage0_results.csv"
mkdir -p "$(dirname "$OUTPUT_CSV")"

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

    # CPU 参考 (仅小尺寸: M*N*K <= 1024^3 = 1G)
    if [ $((M * N * K)) -le 1073741824 ]; then
        echo "  [Reference] CPU..."
        cpu_out=$(run_benchmark "$M" "$N" "$K" "cpu")
        cpu_time=$(echo "$cpu_out" | sed -n '1p')
        cpu_gflops=$(echo "$cpu_out" | sed -n '2p')
        if [ -n "$cublas_gflops" ] && [ "$cublas_gflops" != "0" ]; then
            rel_cpu=$(echo "scale=2; $cpu_gflops * 100 / $cublas_gflops" | bc)
        else
            rel_cpu="0.00"
        fi
        [[ $rel_cpu == .* ]] && rel_cpu="0$rel_cpu"
        echo "$cfg_name,$M,$N,$K,cpu,NA,NA,NA,NA,NA,$cpu_time,$cpu_gflops,${rel_cpu}%" >> "$OUTPUT_CSV"
        echo "    CPU: time=${cpu_time}ms, GFLOPS=${cpu_gflops}, rel=${rel_cpu}%"
    fi

    echo ""

done < "$CONFIGS"

echo "=========================================="
echo "Results saved to: $OUTPUT_CSV"
echo "=========================================="
cat "$OUTPUT_CSV"
