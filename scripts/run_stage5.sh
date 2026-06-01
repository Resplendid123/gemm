#!/bin/bash

# Stage 5: Double Buffering 与访存流水化
# 比较 double-buffer 版本与 cuBLAS 的性能差异

set -euo pipefail

BENCH="./build/benchmark"
CONFIGS="./configs/stage.txt"
RUNS=5

OUTPUT_CSV="./results/stage5_results.csv"
mkdir -p "$(dirname "$OUTPUT_CSV")"

# Double Buffering 配置: config_id|BM|BN|BK|TM|TN
DB_CONFIGS=(
    "1|64|64|8|4|4"      # Config A: 小块配置
    "2|64|128|8|4|8"     # Config B: 大列配置
    "3|128|128|8|8|8"    # Config C: 大块配置
    "4|128|128|16|8|8"   # Config D: 大块配置
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
        local gflops=$(echo "$out" | grep -oP 'GFLOPS: \K[0-9.]+')

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

    # 测试所有 double buffering 配置
    for db_cfg in "${DB_CONFIGS[@]}"; do
        IFS='|' read -r config_id BM BN BK TM TN <<< "$db_cfg"
        echo "  [Stage5] Double Buffering (config=$config_id: BM=$BM, BN=$BN, BK=$BK, TM=$TM, TN=$TN)..."

        db_out=$(run_benchmark "$M" "$N" "$K" "doublebuf" "--config=$config_id")
        db_time=$(echo "$db_out" | sed -n '1p')
        db_gflops=$(echo "$db_out" | sed -n '2p')

        # 计算相对于 cuBLAS 的性能百分比
        if [ -n "$cublas_gflops" ] && [ "$cublas_gflops" != "0" ]; then
            rel_cublas=$(echo "scale=2; $db_gflops * 100 / $cublas_gflops" | bc)
        else
            rel_cublas="0.00"
        fi
        [[ $rel_cublas == .* ]] && rel_cublas="0$rel_cublas"

        echo "$cfg_name,$M,$N,$K,doublebuf,$BM,$BN,$BK,$TM,$TN,$db_time,$db_gflops,${rel_cublas}%" >> "$OUTPUT_CSV"
        echo "    Double Buffer: time=${db_time}ms, GFLOPS=${db_gflops}, rel=${rel_cublas}%"
    done
    echo ""

done < "$CONFIGS"

echo "=========================================="
echo "Results saved to: $OUTPUT_CSV"
echo "=========================================="
cat "$OUTPUT_CSV"
