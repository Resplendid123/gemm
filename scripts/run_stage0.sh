#!/bin/bash

BENCH="./build/benchmark"
CONFIGS="./configs/stage0.txt"
BLOCK_SIZE_X=16
BLOCK_SIZE_Y=16
RUNS=5

OUTPUT_CSV="./results/stage0_results.csv"
mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "config_name,M,N,K,block_x,block_y,kernel,avg_time_ms,avg_gflops,rel_to_cublas_pct,max_relative_error,avg_relative_error,validation_passed" > "$OUTPUT_CSV"

# 运行配置文件中的每一条配置
while IFS= read -r line || [ -n "$line" ]; do
    # 跳过空行和注释
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    # 解析字段: name|M|N|K|validate(yes/no)|tolerance
    IFS='|' read -r cfg_name M N K do_validate tol <<< "$line"
    # 默认块尺寸
    echo "Running config: $cfg_name ($M x $N x $K), validate=$do_validate, tol=$tol"

    declare -A results_time
    declare -A results_gflops
    declare -A results_maxerr
    declare -A results_avgerr
    declare -A results_passed

    # 大规模矩阵不测 CPU；小规模测试 cublas, cpu
    total_elements=$((M * N))
    if [ $total_elements -gt 1048576 ]; then
        kernels_list="cublas"
        echo "  [策略] 大规模矩阵 - 跳过 CPU 测试"
    else
        kernels_list="cublas cpu"
    fi

    # 迭代内核
    for kernel in $kernels_list; do
        echo "  Testing $kernel..."
        total_time=0
        total_gflops=0
        for i in $(seq 1 $RUNS); do
            out=$($BENCH $M $N $K $kernel $BLOCK_SIZE_X $BLOCK_SIZE_Y 2>&1)
            time=$(echo "$out" | grep -oP '[0-9.]+(?= ms)' | head -1)
            gflops=$(echo "$out" | grep -oP 'GFLOPS: \K[0-9.]+' )
            total_time=$(echo "$total_time + ${time:-0}" | bc)
            total_gflops=$(echo "$total_gflops + ${gflops:-0}" | bc)
        done

        avg_time=$(echo "scale=6; $total_time / $RUNS" | bc)
        avg_gflops=$(echo "scale=6; $total_gflops / $RUNS" | bc)
        # 格式化输出
        avg_time_fmt=$(printf "%.3f" "$avg_time")
        avg_gflops_fmt=$(printf "%.2f" "$avg_gflops")
        results_time["$kernel"]=$avg_time_fmt
        results_gflops["$kernel"]=$avg_gflops_fmt

        # 验证
        if [[ "$do_validate" == "yes" ]]; then
            out_val=$($BENCH $M $N $K $kernel $BLOCK_SIZE_X $BLOCK_SIZE_Y --validate --ref=auto 2>&1)
            max_err=$(echo "$out_val" | grep -oP 'Max relative error: \K[0-9.eE-]+' )
            avg_err=$(echo "$out_val" | grep -oP 'Avg relative error: \K[0-9.eE-]+' )
            passed=$(echo "$out_val" | grep -oP 'Validation passed: \K[01]')
            [ -z "$max_err" ] && max_err="skipped"
            [ -z "$avg_err" ] && avg_err="skipped"
            [ -z "$passed" ] && passed=0
        else
            max_err="skipped"
            avg_err="skipped"
            passed=1
        fi

        results_maxerr["$kernel"]=$max_err
        results_avgerr["$kernel"]=$avg_err
        results_passed["$kernel"]=$passed
    done

    # 计算相对于 cuBLAS 的百分比并写入 CSV
    cublas_gflops=${results_gflops["cublas"]}
    for kernel in $kernels_list; do
        # 如果该内核未运行（跳过 cpu），跳过写入
        if [ -z "${results_time["$kernel"]}" ]; then
            continue
        fi

        avg_time=${results_time["$kernel"]}
        avg_gflops=${results_gflops["$kernel"]}
        max_err=${results_maxerr["$kernel"]}
        avg_err=${results_avgerr["$kernel"]}
        passed=${results_passed["$kernel"]}

        if [ "$kernel" == "cublas" ]; then
            rel="100.00"
        else
            # 计算百分比，保留两位小数
            if [ -n "$cublas_gflops" ] && [ "$cublas_gflops" != "0" ]; then
                rel=$(echo "scale=2; $avg_gflops * 100 / $cublas_gflops" | bc)
            else
                rel="0.00"
            fi
            if [[ $rel == .* ]]; then
                rel="0$rel"
            fi
        fi

        # 对于 cuBLAS 与 CPU，不记录 block size（留空）；其他内核记录默认块尺寸
        if [ "$kernel" == "cublas" ] || [ "$kernel" == "cpu" ]; then
            block_x_field="NA"
            block_y_field="NA"
        else
            block_x_field="$BLOCK_SIZE_X"
            block_y_field="$BLOCK_SIZE_Y"
        fi

        echo "$cfg_name,$M,$N,$K,$block_x_field,$block_y_field,$kernel,$avg_time,$avg_gflops,$rel%,$max_err,$avg_err,$passed" >> "$OUTPUT_CSV"
    done

done < "$CONFIGS"

echo -e "\n结果已保存到: $OUTPUT_CSV"
cat "$OUTPUT_CSV"