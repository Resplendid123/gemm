#!/bin/bash

set -euo pipefail

BENCH="./build/benchmark"
CONFIGS="./configs/stage3.txt"
RUNS=5

OUTPUT_CSV="./results/stage3_results.csv"
mkdir -p "$(dirname "$OUTPUT_CSV")"

# Register blocking 配置: config_id|BM|BN|BK|TM|TN
REGISTER_CONFIGS=(
    "1|64|64|8|4|4"        # Config A: 小块配置
    "2|64|128|8|4|8"       # Config B: 大列配置
    "3|128|128|8|8|8"      # Config C: 大块配置 BK=8
    "4|128|128|16|8|8"     # Config D: 大块配置 BK=16
)

echo "config_name,M,N,K,BM,BN,BK,TM,TN,kernel,avg_time_ms,avg_gflops,rel_to_cublas_pct" > "$OUTPUT_CSV"

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
    echo "Running config: $cfg_name ($M x $N x $K)"

    # cuBLAS 基准
    echo "  Testing cublas..."
    cublas_out=$(run_benchmark "$M" "$N" "$K" "cublas")
    cublas_time=$(echo "$cublas_out" | sed -n '1p')
    cublas_gflops=$(echo "$cublas_out" | sed -n '2p')
    echo "$cfg_name,$M,$N,$K,NA,NA,NA,NA,NA,cublas,$cublas_time,$cublas_gflops,100.00%" >> "$OUTPUT_CSV"

    # 测试三种 register blocking 配置
    for reg_cfg in "${REGISTER_CONFIGS[@]}"; do
        IFS='|' read -r config_id BM BN BK TM TN <<< "$reg_cfg"
        echo "  Testing register (config=$config_id: BM=$BM, BN=$BN, BK=$BK, TM=$TM, TN=$TN)..."

        reg_out=$(run_benchmark "$M" "$N" "$K" "register" "--config=$config_id")
        reg_time=$(echo "$reg_out" | sed -n '1p')
        reg_gflops=$(echo "$reg_out" | sed -n '2p')

        # 计算相对于 cuBLAS 的性能百分比
        if [ -n "$cublas_gflops" ] && [ "$cublas_gflops" != "0" ]; then
            rel_cublas=$(echo "scale=2; $reg_gflops * 100 / $cublas_gflops" | bc)
        else
            rel_cublas="0.00"
        fi
        [[ $rel_cublas == .* ]] && rel_cublas="0$rel_cublas"

         echo "$cfg_name,$M,$N,$K,$BM,$BN,$BK,$TM,$TN,register,$reg_time,$reg_gflops,${rel_cublas}%" >> "$OUTPUT_CSV"
    done
    echo ""  # 添加空行分隔不同配置的输出

done < "$CONFIGS"

echo -e "\n结果已保存到: $OUTPUT_CSV"
echo ""
echo "=== Results Summary ==="
cat "$OUTPUT_CSV"