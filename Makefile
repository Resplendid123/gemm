# 编译器
NVCC = nvcc
# 编译选项：优化级别 O3，C++17 标准，指定 GPU 架构
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_86 -Iinclude -lineinfo
# 链接库：cublas（矩阵运算库），cudart（CUDA 运行时）
LDFLAGS = -lcublas -lcudart

# 源文件列表
SRC = src/main.cu src/cpu_reference.cpp src/cublas_reference.cu src/kernels/naive.cu src/kernels/shared_memory.cu src/kernels/register_blocked.cu src/kernels/bank_conflict.cu src/kernels/double_buffered.cu src/kernels/tensor_core.cu src/validate.cpp

# 目标可执行文件
TARGET = build/benchmark

M ?= 4096
N ?= 4096
K ?= 4096
KERNEL ?= tensor
CONFIG ?= 2

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

# 阶段 7 完整 profiling：保存为 ncu_*.ncu-rep（已纳入仓库）
ncu-report: $(TARGET)
	@mkdir -p build
	@echo "Profiling register / bank / doublebuf / tensor at 1024 and 4096..."
	cd build && \
	  ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats \
	      --kernel-name regex:register_blocking --launch-skip 1 --launch-count 1 \
	      -o ncu_register_1024  ../$(TARGET) 1024 1024 1024 register --config=3 && \
	  ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats \
	      --kernel-name regex:bank_conflict    --launch-skip 1 --launch-count 1 \
	      -o ncu_bank_1024     ../$(TARGET) 1024 1024 1024 bank    --config=3 && \
	  ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats \
	      --kernel-name regex:double_buffer   --launch-skip 1 --launch-count 1 \
	      -o ncu_doublebuf_1024 ../$(TARGET) 1024 1024 1024 doublebuf --config=4 && \
	  ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats \
	      --kernel-name regex:register_blocking --launch-skip 1 --launch-count 1 \
	      -o ncu_register_4096 ../$(TARGET) 4096 4096 4096 register --config=3 && \
	  ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats \
	      --kernel-name regex:bank_conflict    --launch-skip 1 --launch-count 1 \
	      -o ncu_bank_4096     ../$(TARGET) 4096 4096 4096 bank    --config=4 && \
	  ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats \
	      --kernel-name regex:double_buffer   --launch-skip 1 --launch-count 1 \
	      -o ncu_doublebuf_4096 ../$(TARGET) 4096 4096 4096 doublebuf --config=4 && \
	  ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats \
	      --kernel-name regex:wmma_gemm_fp32  --launch-skip 1 --launch-count 1 \
	      -o ncu_tensor_4096   ../$(TARGET) 4096 4096 4096 tensor  --config=2
	@echo "Done. Reports in build/ncu_*.ncu-rep"
	@echo "Convert to CSV: ncu --import build/ncu_*.ncu-rep --csv --page raw"

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

validate: $(TARGET)
	./build/benchmark $(M) $(N) $(K) $(KERNEL) --config=$(CONFIG) --validate

validate-all: $(TARGET)
	@echo "Running validation for all kernels..."
	@for kernel in naive shared register bank doublebuf tensor; do \
		echo "Validating kernel: $$kernel..."; \
		./build/benchmark $(M) $(N) $(K) $$kernel --config=$(CONFIG) --validate; \
	done

# 运行所有 stage
run-all: $(TARGET)
	@mkdir -p results
	@echo "=========================================="
	@echo "Running all stages..."
	@echo "=========================================="
	@bash scripts/run_stage0.sh
	@bash scripts/run_stage1.sh
	@bash scripts/run_stage2.sh
	@bash scripts/run_stage3.sh
	@bash scripts/run_stage4.sh
	@bash scripts/run_stage5.sh
	@bash scripts/run_stage6.sh
	@echo "=========================================="
	@echo "All stages completed!"
	@echo "Results saved to: results/"
	@echo "=========================================="

# 声明伪目标
.PHONY: all clean bank-conflict run validate validate-all run-all ncu-report
