# GEMM CUDA 优化实验报告

## 实验基本信息

- **实验主题**: 矩阵乘法算子的实现、优化与体系结构分析
- **实验周期**: 2026 年 5 月 26 日 — 2026 年 6 月 26 日
- **目标分数**: 完成阶段 7 (Profiling、Roofline 建模与整体总结)，得 85 分

---

## 阶段 0: 实验准备与性能基线

### 0.1 实现概述

| Kernel   | 描述                                     |
| -------- | ---------------------------------------- |
| Kernel 0 | CPU reference GEMM，行主序实现，用于验证 |
| cuBLAS   | NVIDIA cuBLAS 库作为性能参考基准         |

### 0.2 实现内容

1. **CPU Reference GEMM**: 实现标准三重循环矩阵乘法作为正确性基准
2. **CUDA Event 计时器**: 使用 `cudaEventCreate`/`cudaEventElapsedTime` 精确测量 kernel 执行时间
3. **cuBLAS Baseline**: 调用 `cublasSgemm` 作为性能参考基准
4. **统一 Benchmark 程序**: 支持命令行参数配置 M/N/K/kernel type，支持 `--validate` 验证

### 0.3 性能计算

$$\text{GFLOPS} = \frac{2 \times M \times N \times K}{t_{sec} \times 10^9}$$

### 0.4 输出指标

- kernel 平均运行时间 (ms)
- GFLOPS
- 相对 cuBLAS 的性能比例 (%)
- 正确性误差（最大相对误差、平均相对误差）

---

## 阶段 1: 朴素 CUDA GEMM

### 1.1 实现概述

每个 thread 负责计算输出矩阵 C 的一个元素，通过线程索引直接映射到输出位置。

### 1.2 核心代码

```c
__global__ void naive_gemm_kernel(float *C, const float *A, float *B,
                                  int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}
```

### 1.3 回答问题

**Q1: 为什么 naive kernel 性能较差？**

> - 每次计算 C[i][j] 需要从全局内存读取 A 的一行 (K 次) 和 B 的一列 (K 次)
> - 相邻线程的 A 访问是连续的，但是B不是
> - 总共需要读取 2×M×N×K 个元素，但 A/B 的每个元素在计算不同输出时被重复读取
> - 全局内存访问延迟高，带宽利用率低

**Q2: A 和 B 的访存模式分别是什么？**

> - **A 矩阵**: 按行访问，`A[row * K + k]` 是连续的全局内存访问
> - **B 矩阵**: 按列访问，`B[k * N + col]` 访问跳距为 N（不连续）

**Q3: 哪些访问是连续的，哪些不是？**

> - **A 访问**: 连续访问（同一个 warp 内的线程访问同一行的连续元素）
> - **B 访问**: 非连续访问（同一个 warp 内的线程访问间隔为 N 的元素）

**Q4: 每个 A/B 元素被复用了多少次？**

> - **A 元素 A[row][k]**: 被用于计算 C[row][0] 到 C[row][N-1]，即 N 次
> - **B 元素 B[k][col]**: 被用于计算 C[0][col] 到 C[M-1][col]，即 M 次
> - 由于没有缓存机制，每次计算都需重新从全局内存读取，没用复用元素

**Q5: 该 kernel 更接近 memory-bound 还是 compute-bound？**

> - **Memory-bound**
> - 计算量: 2×M×N×K 次浮点乘加
> - 内存访问量: M×N×K (A) + K×N (B) + M×N (C) ≈ 2×M×N×K *4B (FP32)
> - 算术强度 = 计算量/内存量 ≈ 0.25 FLOP/byte，Memory Bound

### 1.4 不同 Block Size 性能对比

测试了 5 种不同的 block 配置：(8×8), (16×16), (32×8), (8×32), (32×32)

#### 1024×1024×1024 测试结果

| Block Size | 时间 (ms) | GFLOPS | 相对 cuBLAS |
| ---------- | --------- | ------ | ----------- |
| 8×8        | 5.743     | 373.95 | 9.79%       |
| 16×16      | 4.653     | 461.55 | 12.09%      |
| 32×8       | 4.590     | 467.83 | 12.25%      |
| 8×32       | 5.732     | 374.67 | 9.81%       |
| 32×32      | 4.968     | 432.23 | 11.32%      |

#### 4096×4096×4096 测试结果

| Block Size | 时间 (ms) | GFLOPS | 相对 cuBLAS |
| ---------- | --------- | ------ | ----------- |
| 8×8        | 354.915   | 387.25 | 8.17%       |
| 16×16      | 256.531   | 535.79 | 11.31%      |
| 32×8       | 260.273   | 528.07 | 11.15%      |
| 8×32       | 307.036   | 447.64 | 9.45%       |
| 32×32      | 261.551   | 525.49 | 11.09%      |

### 1.5 Block Size 分析

**最优配置: 16×16 和 32×8**

> - **16×16 (256 threads/block)**: 性能最优，线程数量适中，occupancy 较好
> - **32×8**: 性能与 16×16 接近，因为 warp 大小为 32，32×8 恰好对应 8 个 warp
> - **8×8 (64 threads)**: 线程数太少，occupancy 低，性能较差
> - **8×32**: 性能较差，因为 warp 内的线程访问 B 矩阵的列间隔不连续
> - **32×32 (1024 threads)**: 线程数过多，寄存器压力增大，部分配置下反而下降

**关键发现**: Block size 的选择应保证 warp 对齐（避免半个 warp 的情况），同时考虑线程数和寄存器压力的平衡。

---

## 阶段 2: Shared Memory 分块 GEMM

### 2.1 实现概述

将输出矩阵划分为多个 tile，每个 block 负责一个 tile。沿 K 维度循环时，将 A 和 B 的局部数据加载到共享内存，block 内线程复用共享内存中的数据。

实现支持三种 tile size：8×8、16×16、32×32，根据 block size 自动选择对应版本的 kernel。

### 2.2 核心代码

```c
#define TILE_SIZE 16
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

// TILE_SIZE=8/16/32 各有一个版本，以下为 16x16 版本示例
__global__ void shared_memory_gemm_kernel_16(float *C, const float *A, const float *B,
                                              int M, int N, int K,
                                              float alpha, float beta) {
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    float acc = 0.0f;

    // 沿 K 维度分块循环
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // 加载 A tile 到共享内存
        int A_row = block_row * TILE_SIZE + row;
        int A_col = tile * TILE_SIZE + col;
        if (A_row < M && A_col < K)
            As[row][col] = A[A_row * K + A_col];
        else
            As[row][col] = 0.0f;

        // 加载 B tile 到共享内存
        int B_row = tile * TILE_SIZE + row;
        int B_col = block_col * TILE_SIZE + col;
        if (B_row < K && B_col < N)
            Bs[row][col] = B[B_row * N + B_col];
        else
            Bs[row][col] = 0.0f;

        __syncthreads();

        // 计算当前 tile 的贡献
        for (int k = 0; k < TILE_SIZE; ++k)
            acc += As[row][k] * Bs[k][col];

        __syncthreads();
    }

    // 写回结果
    int C_row = block_row * TILE_SIZE + row;
    int C_col = block_col * TILE_SIZE + col;
    if (C_row < M && C_col < N)
        C[C_row * N + C_col] = alpha * acc + beta * C[C_row * N + C_col];
}
```

### 2.3 不同 Tile Size 性能对比

测试了三种 tile size：8×8、16×16、32×32

#### 1024×1024×1024 测试结果

| Tile Size | 时间 (ms) | GFLOPS | 相对 cuBLAS |
| --------- | --------- | ------ | ----------- |
| 8×8       | 4.596     | 467.30 | 13.25%      |
| 16×16     | 3.501     | 613.32 | 17.39%      |
| 32×32     | 4.072     | 527.41 | 14.95%      |

#### 4096×4096×4096 测试结果

| Tile Size | 时间 (ms) | GFLOPS | 相对 cuBLAS |
| --------- | --------- | ------ | ----------- |
| 8×8       | 289.241   | 475.26 | 9.99%       |
| 16×16     | 188.595   | 728.94 | 15.32%      |
| 32×32     | 200.124   | 686.77 | 14.44%      |

### 2.4 Tile Size 分析

> - **8×8**: 共享内存占用最小 (2×8×8×4B = 512B)，但 threads 数量少 (64)，occupancy 较低
> - **16×16**: 平衡配置，共享内存占用 2KB，threads 256 个，是常用配置
> - **32×32**: 共享内存占用最大 (2×32×32×4B = 8KB)，threads 1024 个，寄存器压力增大

**三种 tile size 性能相近**，原因是当前实现的瓶颈不在共享内存大小，而在：
- 每个 tile 仍需从全局内存读取一次
- K 维度循环次数减少但单次计算量增加
- 当前实现仍是 memory-bound

### 2.5 cuBLAS vs Naive vs Shared Memory 性能对比

综合各配置下最优 block/tile size，对比 cuBLAS、Naive、Shared Memory 三种实现的性能。

| 配置           | cuBLAS GFLOPS | Naive GFLOPS   | Shared GFLOPS | Naive 相对 cuBLAS | Shared 相对 cuBLAS |
| -------------- | ------------- | -------------- | ------------- | ----------------- | ------------------ |
| 256×256        | 382.71        | 407.78 (32×8)  | 565.17 (32×8) | 106.55%           | 147.68%            |
| 1024×1024      | 3817.00       | 467.83 (32×8)  | 677.63 (32×8) | 12.25%            | 17.75%             |
| 4096×4096      | 4736.05       | 535.79 (16×16) | 843.43 (32×8) | 11.31%            | 17.80%             |
| 4096×1024×8192 | 5279.91       | 532.56 (16×16) | 803.94 (32×8) | 10.09%            | 15.23%             |
| 8192×512×4096  | 4671.76       | 472.30 (32×8)  | 684.40 (32×8) | 10.11%            | 14.65%             |
| 1000×1000      | 4163.03       | 462.66 (8×32)  | 648.41 (32×8) | 11.11%            | 15.58%             |
| 511×1023×2047  | 3505.90       | 465.84 (32×8)  | 671.95 (32×8) | 13.29%            | 19.16%             |

#### 性能分析

1. **小矩阵 (256×256)**: **Shared Memory > Naive > cuBLAS**
   - Naive (16×16) 比 cuBLAS 快 46%，Shared Memory 又比 Naive 快 26%
   - 原因：小矩阵下 GPU 并行优势明显，且 cuBLAS 调度开销相对较大

2. **中大矩阵**: cuBLAS 保持 **6-8x** 性能优势
   - Shared Memory 比 Naive 快约 **30%**，主要来源于减少全局内存重复访问

3. **与 cuBLAS 差距原因**:
   - 算术强度仍偏低（~0.25 FLOP/byte）

---

### 2.6 回答问题

**Q1: shared memory tiling 为什么能带来性能提升？**

> - 共享内存带宽是全局内存的 **10-20 倍**
> - 一个 tile 的 A/B 数据只需从全局内存读取 **一次**，之后被同一 block 内的所有线程复用
> - 减少了大量重复的全局内存访问

**Q2: A tile 与 B tile 的数据复用分别体现在哪里？**

> - **A tile**: 同一行元素被 block 内所有线程在计算不同列时复用
> - **B tile**: 同一列元素被 block 内所有线程在计算不同行时复用
> - 沿 K 维度累加时，同一个 tile 的数据被使用 TILE_SIZE 次

**Q3: tile size 增大为什么不一定总是更快？**

> - **共享内存占用增加**: 更大的 tile 占用更多共享内存，减少可同时运行的 block 数量 → **Occupancy 下降**
> - **寄存器压力**: 更大的 tile 可能导致寄存器溢出到 local memory
> - **Bank conflict**: 更大的共享内存数组可能产生更多访问冲突

**Q4: shared memory 占用如何影响 occupancy？**

> - 每个 SM 的共享内存总量有限（如 Ampere 架构最多 167KB per SM）
> - Tile size 增加 → 每个 block 占用更多共享内存 → 可同时运行的 block 数量减少 → **Occupancy 下降**
> - 例如：32×32 tile 占用 8KB，两个 tile 就占满 16KB

**Q5: 优化后 global memory 重复读取是否减少？如何证明？**

> - **减少了**
> - **Naive 版本**: 每个 A 元素被读取 N 次（计算 C[i][0] 到 C[i][N-1] 时重复读取 A[i][k]）
> - **Tiled 版本**: 每个 A 元素仅被读取 K/TILE_SIZE 次（每个 tile 只读取一次）
> - 可通过 Nsight Compute 观察 `gld_transactions` 指标验证

---

## 阶段 3: Register Blocking GEMM

### 3.1 实现概述

Register Blocking 是 Shared Memory Tiling 的进一步优化。在 Shared Memory Tiling 中，每个线程处理输出矩阵的一个元素；而在 Register Blocking 中，每个线程使用寄存器缓存一小块数据（TM×TN 个元素），从而在计算过程中减少对共享内存的访问。

**关键思想**：
- 每个线程负责计算输出矩阵的一个 **TM×TN** 大小的子块
- 将 BK 维度的数据预加载到寄存器中，循环累加
- 减少对共享内存的访问次数，提高数据复用

### 3.2 参数设计

| 配置     | BM  | BN  | BK  | TM  | TN  | 线程数/Block | 共享内存 |
| -------- | --- | --- | --- | --- | --- | ------------ | -------- |
| Config 1 | 64  | 64  | 8   | 4   | 4   | 256          | 16KB     |
| Config 2 | 64  | 128 | 8   | 4   | 8   | 256          | 32KB     |
| Config 3 | 128 | 128 | 16  | 8   | 8   | 256          | 32KB     |

### 3.3 核心代码

```c
template <int BM, int BN, int BK, int TM, int TN>
__global__ void register_blocking_gemm_kernel(...) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 每个线程负责 TM×TN 个输出元素
    float acc[TM][TN];
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    for (int tile = 0; tile < numTiles; ++tile) {
        // 加载 A tile 到共享内存 (每个线程加载 TM×BK 个元素)
        for (int i = 0; i < TM; ++i)
            for (int k = 0; k < BK; ++k)
                As[thread_row_in_block * TM + i][k] = A[...];

        // 加载 B tile 到共享内存 (每个线程加载 BK×TN 个元素)
        for (int j = 0; j < TN; ++j)
            for (int k = 0; k < BK; ++k)
                Bs[k][thread_col_in_block * TN + j] = B[...];

        __syncthreads();

        // 计算：使用寄存器中的数据
        for (int k = 0; k < BK; ++k)
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += As[thread_row_in_block * TM + i][k] *
                                 Bs[k][thread_col_in_block * TN + j];

        __syncthreads();
    }

    // 写回结果
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            C[...] = alpha * acc[i][j] + beta * C[...];
}
```

### 3.4 不同配置性能对比

#### 1024×1024×1024 测试结果

| 配置     | BM×BN×BK   | TM×TN | 时间 (ms) | GFLOPS  | 相对 cuBLAS |
| -------- | ---------- | ----- | --------- | ------- | ----------- |
| cuBLAS   | -          | -     | 0.570     | 3780.34 | 100.00%     |
| Config 1 | 64×64×8    | 4×4   | 3.016     | 712.11  | 18.83%      |
| Config 2 | 64×128×8   | 4×8   | 3.970     | 540.94  | 14.30%      |
| Config 3 | 128×128×16 | 8×8   | 2.929     | 733.24  | 19.39%      |

#### 4096×4096×4096 测试结果

| 配置     | BM×BN×BK   | TM×TN | 时间 (ms) | GFLOPS  | 相对 cuBLAS |
| -------- | ---------- | ----- | --------- | ------- | ----------- |
| cuBLAS   | -          | -     | 28.731    | 4783.90 | 100.00%     |
| Config 1 | 64×64×8    | 4×4   | 150.305   | 914.40  | 19.11%      |
| Config 2 | 64×128×8   | 4×8   | 173.751   | 791.01  | 16.53%      |
| Config 3 | 128×128×16 | 8×8   | 115.003   | 1195.11 | 24.98%      |

### 3.5 配置分析

**Config 3 (128×128×16, 8×8) 表现最优**：
- 更大的 BM×BN 和 BK 可以更好地分摊加载开销
- 每个线程处理 64 个输出元素，寄存器利用率高
- 共享内存占用 32KB，在大多数 GPU 的限制内
- 在 4096×4096 大矩阵下达到 **1195 GFLOPS**，相对 cuBLAS 24.98%

**Config 1 vs Config 2**：
- Config 2 的 BN 更大，对窄矩阵（如 8192×512）更友好
- 但对于方阵，Config 1 表现更稳定

**小矩阵 (256×256)**：
- Config 1 (64×64) 表现最好，512×512 以上大矩阵 Config 3 更优
- 配置过大的 block size 在小矩阵上反而表现不佳

### 3.6 Register Blocking vs Shared Memory 性能对比

| 配置           | cuBLAS GFLOPS | Shared GFLOPS | Register GFLOPS | Shared 相对 cuBLAS | Register 相对 cuBLAS |
| -------------- | ------------- | ------------- | --------------- | ------------------ | -------------------- |
| 256×256        | 376.98        | 565.17        | 438.77          | 149.92%            | 116.39%              |
| 1024×1024      | 3780.34       | 677.63        | 733.24          | 17.92%             | 19.39%               |
| 4096×4096      | 4783.90       | 843.43        | 1195.11         | 17.63%             | 24.98%               |
| 4096×1024×8192 | 5250.76       | 803.94        | 982.10          | 15.31%             | 18.70%               |
| 8192×512×4096  | 4701.46       | 684.40        | 928.82          | 14.56%             | 19.75%               |
| 1000×1000      | 3879.90       | 648.41        | 702.99          | 16.71%             | 18.11%               |
| 511×1023×2047  | 3208.63       | 671.95        | 756.14          | 20.94%             | 23.56%               |

**观察**：
- Config 3 (128×128×16, 8×8) 在大多数配置下表现最优，性能提升明显
- Register Blocking 相比 Shared Memory 有一定提升，尤其在 4096×4096 大矩阵下达到 24.98%

### 3.7 回答问题

**Q1: 为什么 thread-level tiling 能进一步提升性能？**

> - **增加计算密度**：每个线程计算 TM×TN 个输出元素，相比单元素计算，数据加载开销被分摊到更多计算上
> - **减少线程同步开销**：每个线程完成更多工作，减少了 `__syncthreads()` 的调用频率
> - **提高指令级并行**：单线程内的计算更密集，编译器可以更好地调度指令，隐藏延迟
> - **更好地利用寄存器**：TM×TN 个累加器可以更好地利用 GPU 的大量寄存器，避免资源闲置

**Q2: 一个 thread 计算多个输出元素时，A/B 数据如何在寄存器中复用？**

> - **A 数据的复用**：每个线程负责 TM 行输出，A 数据的一行（BK 个元素）被加载到共享内存后，需要广播给所有 TN 列的计算
> - **B 数据的复用**：每个线程负责 TN 列输出，B 数据的一列（BK 个元素）被加载到共享内存后，需要广播给所有 TM 行的计算
> - **内层循环复用**：在 `for (int k = 0; k < BK; ++k)` 循环中，每一对 A[k] 和 B[k] 被用于更新整个 TM×TN 的累加器矩阵
> - **复用次数**：每个 A/B 元素在一个 tile 内被复用 TM×TN 次（如果 TM=TN=8，则复用 64 次）

**Q3: 为什么 shared memory 访问次数可能减少？**

> - **数据共享**：在 shared memory tiling 中，每个 A 行/B 列元素只被一个线程使用；在 register blocking 中，同一行的 A 数据可以被 TN 个线程共享（通过共享内存广播）
> - **减少冗余加载**：每个线程加载 A 数据的 TM 行、B 数据的 TN 列，但通过共享内存的广播机制，避免了重复加载相同数据
> - **BK 增大效果**：BK 越大，每个 tile 的计算量越大，加载开销被分摊到更多计算上，相对加载次数减少
> - **加载模式优化**：通过将 TM×BK 个 A 元素和 BK×TN 个 B 元素打包加载，减少了总的加载次数

**Q4: 为什么寄存器过多会降低 occupancy？**

> - **寄存器数量有限**：每个 SM 的寄存器文件大小有限（如 Ampere 架构约 64KB per SM），分配给每个线程的寄存器数量直接影响可同时运行的线程数
> - **线程级寄存器压力**：TM×TN 个累加器 + 临时变量，每个线程需要大量寄存器。TM×TN=8×8=64 时，仅累加器就需要 64 个 float（256B）寄存器
> - **Occupancy 计算**：可同时运行的线程数 = (每个 SM 寄存器总量) / (每个线程寄存器数)。寄存器过多 → 可运行线程数减少 → **Occupancy 下降**
> - **寄存器溢出**：如果寄存器需求超过硬件限制，数据会被溢出到 local memory（相当于全局内存），性能严重下降
> - **共享内存的替代作用**：当寄存器不足时，数据必须存储到共享内存，增加了访问延迟

**Q5: 当前 kernel 的主要瓶颈是否发生变化？**

> - **仍然是 Memory-bound**：虽然寄存器级分块提高了数据复用，但数据仍需从全局内存 → 共享内存 → 寄存器的路径，瓶颈仍在内存带宽
> - **算术强度变化**：理论上算术强度 = 2×TM×TN×BK / (TM×BK + BK×TN) 个 FLOP/byte。TM=TN=8, BK=16 时约为 1.0 FLOP/byte，仍低于峰值所需
> - **瓶颈转移**：从"全局内存带宽"部分转移到"共享内存带宽 + 全局内存带宽"的组合瓶颈
> - **与 cuBLAS 差距**：cuBLAS 使用 Tensor Core 可以达到 ~100 FLOP/byte，而我们仅用 FP32 CUDA Core，提升空间有限

---

## 阶段 4: Bank Conflict 消除 GEMM

### 4.1 实现概述

基于 Stage 3 的 Register Blocking GEMM，通过在 shared memory 声明中添加 padding 列来避免 bank conflict。Bank conflict 是指 warp 内多个线程同时访问映射到同一个 bank 的不同地址，导致访问串行化的问题。

**关键优化点**：
- 在 A tile 的第二维添加 +1 padding：`As[BM][BK + 1]`
- 在 B tile 的第二维添加 +1 padding：`Bs[BK][BN + 1]`
- Padding 改变了 bank 映射关系，使原本冲突的访问错开

### 4.2 实现方法

**Bank Conflict 分析**：
- CUDA shared memory 被划分为 32 个 bank，每个 bank 宽度为 4 字节（float）
- 同一 warp 内线程访问时，如果多个线程访问同一 bank 的不同地址，会发生 conflict
- 在 GEMM 计算阶段，`Bs[k][col]` 的访问模式最容易产生冲突：
  - 当 k 相同时，所有线程访问同一行的不同列，可能映射到同一 bank

**Padding 优化原理**：
```
原始 Bs[k][col]: bank = (k * BN + col) % 32
Padding 后 Bs[k][col]: bank = (k * (BN+1) + col) % 32
```
- 添加 padding 后，列索引的 bank 映射发生变化
- 原本冲突的访问（相同 bank ID）被错开到不同的 bank
- Padding=1 可以消除大多数 stride-32 的访问冲突

**共享内存声明**：
```c
// Config A/B/C/D 对应的 padding 版本
__shared__ float As1[64][8 + 1];  __shared__ float Bs1[8][64 + 1];
__shared__ float As2[64][8 + 1];  __shared__ float Bs2[8][128 + 1];
__shared__ float As3[128][8 + 1]; __shared__ float Bs3[8][128 + 1];
__shared__ float As4[128][16 + 1]; __shared__ float Bs4[16][128 + 1];
```


### 4.3 不同配置性能对比

#### 1024×1024×1024 测试结果

| 配置     | BM×BN×BK    | TM×TN | 时间 (ms) | GFLOPS  | 相对 cuBLAS |
| -------- | ----------- | ----- | --------- | ------- | ----------- |
| cuBLAS   | -           | -     | 0.570     | 3780.34 | 100.00%     |
| Config 1 | 64×64×8     | 4×4   | 3.016     | 712.11  | 18.83%      |
| Config 2 | 64×128×8    | 4×8   | 3.970     | 540.94  | 14.30%      |
| Config 3 | 128×128×8   | 8×8   | 2.989     | 718.70  | 19.01%      |
| Config 4 | 128×128×16  | 8×8   | 2.929     | 733.24  | 19.39%      |

#### 4096×4096×4096 测试结果

| 配置     | BM×BN×BK    | TM×TN | 时间 (ms) | GFLOPS  | 相对 cuBLAS |
| -------- | ----------- | ----- | --------- | ------- | ----------- |
| cuBLAS   | -           | -     | 28.731    | 4783.90 | 100.00%     |
| Config 1 | 64×64×8     | 4×4   | 150.305   | 914.40  | 19.11%      |
| Config 2 | 64×128×8    | 4×8   | 173.751   | 791.01  | 16.53%      |
| Config 3 | 128×128×8   | 8×8   | 119.876   | 1147.23 | 23.98%      |
| Config 4 | 128×128×16  | 8×8   | 115.003   | 1195.11 | 24.98%      |

### 4.4 Bank Conflict vs Register Blocking 性能对比

| 配置           | cuBLAS GFLOPS | Register GFLOPS | Bank GFLOPS | Register 相对 cuBLAS | Bank 相对 cuBLAS |
| -------------- | ------------- | --------------- | ----------- | -------------------- | ---------------- |
| 256×256        | 400.76        | 438.77          | 452.85      | 109.48%              | 112.99%          |
| 1024×1024      | 3344.20       | 733.24          | 731.22      | 21.92%               | 21.86%           |
| 4096×4096      | 4798.87       | 1195.11         | 1179.15     | 24.90%               | 24.57%           |
| 4096×1024×8192 | 5293.33       | 982.10          | 981.14      | 18.55%               | 18.53%           |
| 8192×512×4096  | 4688.03       | 928.82          | 920.79      | 19.82%               | 19.64%           |
| 1000×1000      | 3986.16       | 702.99          | 697.35      | 17.63%               | 17.49%           |
| 511×1023×2047  | 3553.26       | 756.14          | 749.20      | 21.28%               | 21.08%           |


### 4.5 回答问题

**Q1: 一个 warp 内线程如何访问 shared memory？**

> - **Warp 是执行单位**：一个 warp 包含 32 个线程，它们以 SIMD（单指令多数据）方式同时执行。当 warp 执行 shared memory 访问指令时，所有 32 个线程同时发起访问
> - **合并访问**：如果 warp 内线程访问的是连续且对齐的地址（如 `Bs[k][col]` 到 `Bs[k][col+31]`），这些访问会被合并成较少的 memory transaction，效率较高
> - **独立 bank**：每个线程访问不同的 bank 时，访问可以并行进行，互不干扰
> - **访问延迟**：Shared memory 访问延迟约为 10-20 个周期，但多个 bank 可以同时服务请求

**Q2: 哪些访问模式容易发生 bank conflict？**

> - **同列不同行访问**：`Bs[k1][col]` 和 `Bs[k2][col]`（相同列 col，不同行 k1≠k2）- 这是最典型的 bank conflict 模式，因为相同 bank ID 的列在不同行
> - **stride 访问**：当 warp 内线程以固定步长访问共享内存时，如果步长导致多个线程访问同一个 bank，就会发生 conflict
> - **GEMM 中的典型模式**：
>   - 访问 `Bs[k][col]` 时，`k` 在内层循环变化，`col` 在 warp 内连续
>   - 当 `k` 相同时，所有线程访问同一列的不同行，产生 bank conflict
> - **广播冲突**：当多个线程需要访问同一个 bank 的同一行时（数据广播），会串行化

**Q3: padding 为什么可能有效？**

> - **改变 bank 映射**：添加 padding 后，数组的列索引发生变化。原来访问 `Bs[k][col]` 变为访问 `Bs[k][col]`（但 col 对应的 bank ID 改变了）
> - **错开冲突索引**：假设原来 `Bs[k][c]` 和 `Bs[k+1][c]` 冲突（bank = c % 32），添加 padding 后变为 `Bs[k][c+1]` 和 `Bs[k+1][c+1]`（bank = (c+1) % 32）
> - **原理**：Bank ID 由 address % 32 决定。Padding 改变了 address 的低 5 位，使得原本冲突的访问映射到不同的 bank
> - **有效性条件**：当冲突的线程访问的行号之差恰好等于 bank 数量时，padding=1 可以完美消除冲突

**Q4: padding 会带来什么代价？**

> - **共享内存占用增加**：每个 tile 需要额外存储一列数据。例如 `Bs[32][32+1]` 比 `Bs[32][32]` 多占用 32×4=128 字节
> - **内存带宽浪费**：额外加载的 padding 数据没有实际用途，但仍然占用内存带宽
> - **编译器优化受限**：padding 可能影响编译器的自动向量化和其他优化策略
> - **代码复杂度**：需要显式处理 padding 对索引的影响

**Q5: shared memory 使用量增加后是否影响 occupancy？**

> - **每个 SM 共享内存有限**：例如 Ampere 架构每个 SM 最多 167KB shared memory
> - **Occupancy 计算**：Occupancy = (活跃线程数) / (最大线程数)。共享内存占用增加 → 可同时运行的 block 数减少 → **Occupancy 下降**
> - **寄存器压力**：如果 shared memory 占用过大，可能会与寄存器资源竞争，影响每个 SM 的最大线程数
> - **权衡**：Padding=1 只增加很小一部分共享内存（128 字节），对 occupancy 影响有限。但如果 tile 数量多或配置激进，影响会更大

**Q6: bank conflict 降低后性能是否一定提升？为什么？**

> - **不一定**：Bank conflict 消除后性能提升不是必然的，原因包括：
>   - **瓶颈不在此**：如果主要瓶颈是全局内存带宽或计算资源，减少 bank conflict 的收益很小
>   - **Occupancy 下降**：padding 导致的共享内存增加可能降低 occupancy，部分抵消收益
>   - **访问延迟相对较低**：共享内存延迟本身较低（10-20 周期 vs 全局内存 200+ 周期），优化这部分收益有限
>   - **编译器优化**：CUDA 编译器已经处理了部分 bank conflict
> - **结论**：只有当 bank conflict 是主要瓶颈时，消除它才能带来显著性能提升

**Q7: 当前 kernel 中，A tile 和 B tile 哪一个更可能是主要冲突来源？**

> - **B tile 是主要冲突来源**
> - **A tile 分析**：
>   - 访问模式：`As[row][k]`，row 是 `thread_row_in_block`（线程内固定），k 在循环中变化
>   - Warp 内的访问：`As[r][k]` 到 `As[r][k+31]`（对于 32 个线程，相同 row）
>   - 同一行访问是连续的，bank 映射：`(row * BK + k) % 32`。当 row 固定时，不同 k 的访问会映射到不同的 bank，**冲突较少**
>
> - **B tile 分析**：
>   - 访问模式：`Bs[k][col]`，k 在循环中变化，col 是 `thread_col_in_block * TN + j`（线程内变化）
>   - Warp 内的访问：线程 t 访问 `Bs[k][col_t]`，其中 col_t 在 warp 内连续
>   - 当 k 相同时，所有线程访问同一行的不同列：`Bs[k][col_0]` 到 `Bs[k][col_31]`
>   - Bank 映射：`(k * (BN+1) + col_t) % 32`。当 k 固定时，相邻 col 的访问很可能落在同一个 bank
>   - **冲突严重**
>
> - **结论**：B tile 的访问模式更容易导致 bank conflict，因此是主要冲突来源

## 附录：各阶段性能总结

