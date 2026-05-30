# 编译器
NVCC = nvcc
# 编译选项：优化级别 O3，C++17 标准，指定 GPU 架构
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_86 -Iinclude -lineinfo
# 链接库：cublas（矩阵运算库），cudart（CUDA 运行时）
LDFLAGS = -lcublas -lcudart

# 源文件列表
SRC = src/main.cu src/reference_cpu.cpp src/cublas_wrapper.cu src/kernels/naive.cu src/kernels/shared_memory.cu src/kernels/register_blocking.cu src/kernels/bank_conflict.cu src/validate.cpp
# 目标可执行文件
TARGET = build/benchmark

M ?= 1024
N ?= 1024
K ?= 1024
KERNEL ?= bank
CONFIG ?= 1

# 默认目标：创建 build 目录并编译
all: $(TARGET)

$(TARGET): $(SRC) | build
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

# 创建 build 目录（如果不存在）
build:
	mkdir -p build

# 清理编译产物
clean:
	rm -rf build/*

# Bank conflict profiling，使用 ncu 分析 shared memory bank conflicts
bank-conflict:
	cd build && ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./benchmark $(M) $(N) $(K) $(KERNEL) --config=$(CONFIG)

# 运行测试
run:
	./build/benchmark $(M) $(N) $(K) $(KERNEL) --config=$(CONFIG)

roofline: $(TARGET)
	mkdir -p build
	@echo "Collecting roofline data for kernel: $(KERNEL)..."
	cd build && ncu --set full \
		--force-overwrite \
		-o roofline_$(KERNEL) \
		./benchmark $(M) $(N) $(K) $(KERNEL) --config=$(CONFIG)
	@echo "Report saved to build/roofline_$(KERNEL).ncu-rep"

# 声明伪目标
.PHONY: all clean bank-conflict run
