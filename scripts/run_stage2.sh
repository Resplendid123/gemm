#!/bin/bash

# Stage 2: Shared Memory 阻塞
# 测试不同 BM/BN/BK 配置对性能的影响

set -euo pipefail

BENCH="./build/benchmark"
CONFIGS="./configs/stage.txt"
RUNS=5

OUTPUT_CSV="./results/stage2_results.csv"
mkdir -p "$(dirname "$OUTPUT_CSV")"

# Shared Memory 配置: config_id|BM|BN|BK
SHARED_CONFIGS=(
    "1|8|8|8"       # Config A: 8x8 小块
    "2|16|16|16"    # Config B: 16x16 中块
    "3|32|32|32"    # Config C: 32x32 大块
    "4|32|8|32"     # Config D: 32x8 非方形
    "5|8|32|32"     # Config E: 8x32 非方形
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

    # 测试不同 Shared Memory 配置
    for shared_cfg in "${SHARED_CONFIGS[@]}"; do
        IFS='|' read -r config_id BM BN BK <<< "$shared_cfg"
        echo "  [Stage2] Shared Memory (config=$config_id: BM=$BM, BN=$BN, BK=$BK)..."

        shared_out=$(run_benchmark "$M" "$N" "$K" "shared" "--config=$config_id")
        shared_time=$(echo "$shared_out" | sed -n '1p')
        shared_gflops=$(echo "$shared_out" | sed -n '2p')

        if [ -n "$cublas_gflops" ] && [ "$cublas_gflops" != "0" ]; then
            rel=$(echo "scale=2; $shared_gflops * 100 / $cublas_gflops" | bc)
        else
            rel="0.00"
        fi
        [[ $rel == .* ]] && rel="0$rel"

        # shared: TM=1, TN=1
        echo "$cfg_name,$M,$N,$K,shared,$BM,$BN,$BK,1,1,$shared_time,$shared_gflops,${rel}%" >> "$OUTPUT_CSV"
        echo "    Shared: time=${shared_time}ms, GFLOPS=${shared_gflops}, rel=${rel}%"
    done
    echo ""

done < "$CONFIGS"

echo "=========================================="
echo "Results saved to: $OUTPUT_CSV"
echo "=========================================="
cat "$OUTPUT_CSV"
