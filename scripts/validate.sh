#!/bin/bash

# 验证策略脚本
# 根据矩阵规模和数据类型选择合适的验证方法

validate_result() {
    local M=$1
    local N=$2
    local K=$3
    local kernel=$4
    local tolerance=$5
    
    # 计算总元素数
    local total_elements=$((M * N))
    
    # 判断使用哪种参考标准
    local use_cublas="false"
    local use_cpu="false"
    
    # 大规模矩阵 (> 1024*1024 元素) 使用 cuBLAS
    if [ $total_elements -gt 1048576 ]; then
        use_cublas="true"
        echo "  [策略] 大规模矩阵, 使用 cuBLAS 作为参考"
    else
        use_cpu="true"
        echo "  [策略] 中小规模矩阵, 使用 CPU reference 作为参考"
    fi
    
    # 特殊处理: FP16/TensorCore 允许更大误差
    if [[ "$kernel" == *"tensorcore"* ]] || [[ "$kernel" == *"fp16"* ]]; then
        tolerance=$(echo "$tolerance * 100" | bc)  # 放宽 100 倍
        echo "  [策略] TensorCore/FP16, 放宽误差阈值至 ${tolerance}"
    fi
    
    # 运行验证
    local max_error=""
    if [ "$use_cpu" == "true" ]; then
        # 使用 CPU reference 验证
        out_val=$($BENCH $M $N $K $kernel $BLOCK_SIZE_X $BLOCK_SIZE_Y --validate --ref=cpu 2>&1)
        max_error=$(echo "$out_val" | grep -oP 'Max (relative )?error: \K[0-9.eE-]+')
    else
        # 使用 cuBLAS 验证
        out_val=$($BENCH $M $N $K $kernel $BLOCK_SIZE_X $BLOCK_SIZE_Y --validate --ref=cublas 2>&1)
        max_error=$(echo "$out_val" | grep -oP 'Max (relative )?error: \K[0-9.eE-]+')
    fi
    
    # 检查是否通过
    if [ -n "$max_error" ]; then
        passed=$(echo "$max_error < $tolerance" | bc)
        if [ $passed -eq 1 ]; then
            echo "  [验证] ✓ 通过 (误差: $max_error < $tolerance)"
        else
            echo "  [验证] ✗ 失败 (误差: $max_error >= $tolerance)"
        fi
    else
        max_error="validation_failed"
        echo "  [验证] ✗ 验证执行失败"
    fi
    
    echo "$max_error"
}

# 导出函数供主脚本使用
export -f validate_result