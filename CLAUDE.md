# CUDA GEMM 开发指南

## 文件结构

```
include/gemm_common.h     # 函数声明
src/main.cu               # 主程序
src/kernels/*.cu          # kernel 实现
```

## 添加新 Kernel

### 1. `include/gemm_common.h` - 添加声明

### 2. `src/kernels/kernel.cu` - 实现 kernel

### 3. `src/main.cu` - 添加分支

### 5. `scripts/run_stage{num}.sh` - 编写批量测试脚本


## 编译

```bash
make clean
make
```

## 运行

```bash
make run
```
