# 编译器
NVCC = nvcc
# 编译选项：优化级别 O3，C++17 标准，指定 GPU 架构
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_86 -Iinclude
# 链接库：cublas（矩阵运算库），cudart（CUDA 运行时）
LDFLAGS = -lcublas -lcudart

# 源文件列表
SRC = src/main.cu src/reference_cpu.cpp src/cublas_wrapper.cu src/kernels/naive.cu src/validate.cpp
# 目标可执行文件
TARGET = build/benchmark

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

# 声明伪目标
.PHONY: all clean