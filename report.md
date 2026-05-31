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

| 配置             | cuBLAS GFLOPS | Naive GFLOPS   | Shared GFLOPS | Naive 相对 cuBLAS | Shared 相对 cuBLAS |
| ---------------- | ------------- | -------------- | ------------- | ----------------- | ----------------- |
| 256×256          | 402.53        | 407.78 (32×8)  | 543.38 (8×32) | 101.30%           | 134.98%           |
| 1024×1024        | 3817.00       | 467.83 (32×8)  | 677.27 (8×32) | 12.25%            | 17.71%            |
| 4096×4096        | 4736.05       | 535.79 (16×16) | 855.49 (8×32) | 11.31%            | 18.08%            |
| 4096×1024×8192   | 5279.91       | 532.56 (16×16) | 768.78 (8×32) | 10.09%            | 14.56%            |
| 8192×512×4096    | 4671.76       | 472.30 (32×8)  | 684.48 (8×32) | 10.11%            | 14.64%            |
| 1000×1000        | 4163.03       | 462.66 (8×32)  | 645.63 (8×32) | 11.11%            | 15.49%            |
| 511×1023×2047    | 3505.90       | 465.84 (32×8)  | 671.40 (8×32) | 13.28%            | 19.13%            |

#### 性能分析

1. **小矩阵 (256×256)**: **Shared Memory > Naive > cuBLAS**
   - Shared Memory (8×32) 比 cuBLAS 快 **35%**，表现非常出色
   - 原因：小矩阵下 GPU 并行优势明显，且 cuBLAS 调度开销相对较大

2. **中大矩阵**: cuBLAS 保持 **5-8x** 性能优势
   - Shared Memory 比 Naive 快约 **30-60%**，主要来源于减少全局内存重复访问
   - 最优 tile size 配置从 32×8 变为 8×32，说明针对不同矩阵形状需要选择不同配置

3. **与 cuBLAS 差距原因**:
   - 算术强度仍偏低（~0.25 FLOP/byte）
   - cuBLAS 使用了更复杂的分块策略和 Tensor Core

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
__global__ void register_blocked_gemm_kernel(...) {
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

**128×128, BK=8, TM×TN=8×8 配置表现最优**：
- 更大的 BM×BN 和 BK 可以更好地分摊加载开销
- 每个线程处理 64 个输出元素，寄存器利用率高
- 在 4096×4096 大矩阵下达到 **1239.86 GFLOPS**，相对 cuBLAS 25.74%
- BK=8 配置比 BK=16 更优，因为更小的 BK 减少了寄存器压力，提高了 occupancy

**Config 1 vs Config 2**：
- Config 2 的 BN 更大，对窄矩阵（如 8192×512）更友好
- 但对于方阵，Config 1 表现更稳定

**小矩阵 (256×256)**：
- Config 1 (64×64) 表现最好，512×512 以上大矩阵 Config 3 更优
- 配置过大的 block size 在小矩阵上反而表现不佳

### 3.6 Register Blocking vs Shared Memory 性能对比

| 配置           | cuBLAS GFLOPS | Shared GFLOPS | Register GFLOPS | Shared 相对 cuBLAS | Register 相对 cuBLAS |
| ------------- | ------------- | ------------- | --------------- | ------------------ | -------------------- |
| 256×256       | 341.11        | 543.38        | 466.83 (64×64)  | 159.24%             | 136.80%              |
| 1024×1024     | 3488.05       | 677.27        | 768.60 (128×128)| 19.42%             | 22.03%               |
| 4096×4096     | 4815.93       | 855.49        | 1239.86 (128×128, BK=8)| 17.74%      | 25.71%               |
| 4096×1024×8192| 5262.11       | 768.78        | 1002.32 (128×128, BK=8)| 14.60%      | 19.04%               |
| 8192×512×4096 | 4681.16       | 684.48        | 969.92 (128×128, BK=8)| 14.61%       | 20.71%               |
| 1000×1000     | 3939.43       | 645.63        | 737.10 (128×128, BK=8)| 16.38%      | 18.71%               |
| 511×1023×2047 | 3744.48       | 671.40        | 777.72 (128×128, BK=8)| 17.93%      | 20.76%               |

**观察**：
- **128×128, BK=8, TM×TN=8×8** 配置在大多数大矩阵下表现最优
- Register Blocking 相比 Shared Memory 有显著提升，尤其在 4096×4096 大矩阵下达到 **25.71%**
- 小矩阵 (256×256) 下 Shared Memory 表现更好，因为寄存器复用优势在小规模下不明显

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

| 配置           | cuBLAS GFLOPS | Register GFLOPS | Bank GFLOPS    | Register 相对 cuBLAS | Bank 相对 cuBLAS |
| ------------- | ------------- | --------------- | -------------- | ------------------- | ---------------- |
| 256×256       | 318.84        | 466.83          | 438.27 (64×64) | 146.53%             | 137.45%          |
| 1024×1024     | 3807.59       | 768.60          | 747.49 (128×128, BK=16)| 20.17%      | 19.62%           |
| 4096×4096     | 4726.24       | 1239.86         | 1192.44 (128×128, BK=16)| 26.24%      | 25.23%           |
| 4096×1024×8192| 5263.63       | 1002.32         | 965.58 (128×128, BK=16)| 19.04%      | 18.34%           |
| 8192×512×4096 | 4692.01       | 969.92          | 942.08 (128×128, BK=16)| 20.66%      | 20.07%           |
| 1000×1000     | 3889.68       | 737.10          | 709.31 (128×128, BK=16)| 18.97%      | 18.23%           |
| 511×1023×2047 | 3441.47       | 777.72          | 748.80 (128×128, BK=16)| 22.60%      | 21.75%           |

**观察**：
- Bank Conflict 优化后性能与 Register Blocking 版本相近，部分配置略有下降
- 原因：当前 kernel 的主要瓶颈不在 bank conflict，而在全局内存带宽
- Padding 带来的额外共享内存占用可能略微降低了 occupancy，部分抵消了 bank conflict 优化的收益
- 对于小矩阵 (256×256)，Bank 版本反而比 Register 版本慢，因为小矩阵下 occupancy 影响更显著


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

---

## 阶段 5: Double Buffering 与访存流水化

### 5.1 实现概述

Double Buffering 使用两组共享内存缓冲区，将"下一块 tile 的加载"与"当前 tile 的计算"重叠，隐藏访存延迟。

### 5.2 核心代码

```c
__shared__ float As[2][BM][BK];
__shared__ float Bs[2][BK][BN];

int numTiles = (K + BK - 1) / BK;
int cur_buffer = 0;

// 预加载 tile 0
__syncthreads();
cur_buffer = 1 - cur_buffer;

for (int tile = 0; tile < numTiles; ++tile) {
    // 异步加载下一个 tile
    if (tile + 1 < numTiles) {
        // 加载 As[write_stage], Bs[write_stage]
    }
    
    // 计算当前 tile (使用 read_stage)
    for (int k = 0; k < BK; ++k)
        acc += As[read_stage][...] * Bs[read_stage][...];
    
    if (tile + 1 < numTiles) {
        __syncthreads();
        read_stage = write_stage;
        write_stage = 1 - write_stage;
    }
}
```

### 5.3 性能对比

#### 4096×4096×4096 测试结果

| 配置     | BM×BN×BK   | TM×TN | 时间 (ms) | GFLOPS  | 相对 cuBLAS |
| -------- | ---------- | ----- | --------- | ------- | ----------- |
| cuBLAS   | -          | -     | 31.331    | 4387.38 | 100.00%     |
| Config 1 | 64×64×8    | 4×4   | 167.661   | 819.82  | 18.70%      |
| Config 3 | 128×128×8  | 8×8   | 134.984   | 1018.23 | 23.23%      |
| Config 4 | 128×128×16 | 8×8   | 122.048   | 1126.28 | 25.70%      |

### 5.4 Double Buffering vs Register Blocking 性能对比

| 配置           | cuBLAS GFLOPS | Register GFLOPS | DoubleBuf GFLOPS | 提升比例 |
| ------------- | ------------- | --------------- | ---------------- | -------- |
| 256×256       | 318.55        | 456.49          | 362.71           | -20.54%  |
| 1024×1024     | 3888.96       | 765.89          | 744.52           | -2.79%   |
| 4096×4096     | 4276.65       | 1136.77         | 1126.28          | -0.92%   |
| 4096×1024×8192| 5255.16       | 1015.51         | 1042.67          | +2.67%   |
| 8192×512×4096 | 4647.27       | 969.89          | 943.63           | -2.71%   |

### 5.5 问题分析

**Double Buffering 几乎没有带来性能提升，甚至略有下降！**

**原因分析**：

1. **主要瓶颈不在内存延迟**：当前 kernel 仍然是 global memory bandwidth-bound，计算与加载的 overlap 无法解决带宽瓶颈

2. **共享内存占用翻倍**：Double Buffering 需要两组共享内存缓冲区，减少了可同时运行的 block 数量，降低了 occupancy

3. **Tile 切换开销**：Buffer 切换逻辑引入额外开销

4. **计算密度不足**：即使隐藏了加载延迟，计算本身的速度仍然受限于 global memory 带宽

**结论**：Double Buffering 适用于计算密度较高但访存延迟成为瓶颈的场景。当前 CUDA Core kernel 的瓶颈主要在全局内存带宽，而非延迟，因此收益有限。

### 5.6 回答问题

**Q1: double buffering 试图隐藏什么延迟？**

> - **Global memory 加载延迟**：从 global memory 加载数据到 shared memory 的延迟（200+ 周期）
> - **数据传输时间**：通过 L2 cache 传输数据的时间

**Q2: 为什么需要两组 shared memory buffer？**

> - **避免数据竞争**：当一个 buffer 用于计算时，另一个可以同时加载下一块数据
> - **实现流水化**：两组 buffer 交替使用，形成"加载-计算-加载-计算"的流水线

**Q3: double buffering 增加了多少 shared memory 使用量？**

> - **翻倍**：从 `BM×BK + BK×BN` 增加到 `2×(BM×BK + BK×BN)`
> - 例如 Config 4: 2×(128×16 + 16×128) = 8KB → 16KB

**Q4: 在什么情况下 double buffering 收益更明显？**

> - **计算密度高**：每个 tile 的计算量足够大，能充分利用加载的数据
> - **K 维度 tile 数少**：tile 数越少，切换开销占比越大
> - **Tile size 大**：更大的 tile 意味着更长的加载时间，隐藏延迟的收益更大
> - **当前场景**：以上条件都不满足，因此收益有限

**Q5: double buffering 与 padding 是否会相互影响？**

> - **会**：Double buffering 的额外共享内存占用可能加剧 occupancy 下降
> - **需要权衡**：如果 shared memory 资源紧张，可能需要减小 tile size 以支持 double buffering

**Q6: 如果 double buffering 没有带来性能提升，可能原因是什么？**

> - **瓶颈不在延迟**：当前 kernel 是 bandwidth-bound，不是 latency-bound
> - **带宽瓶颈**：无论是否 overlap，加载速度都受限于全局内存带宽
> - **Occupancy 下降**：额外的共享内存占用降低了并行度
> - **实现问题**：需要使用 `cp.async` 或 `cuda::pipeline` 等异步机制才能真正实现 overlap

---

## 阶段 6: Tensor Core GEMM

### 6.1 实现概述

使用 NVIDIA WMMA (Warp Matrix Multiply Accumulate) API 调用 Tensor Core 硬件单元加速矩阵乘法。

### 6.2 WMMA 配置

```c
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Fragment 声明
wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

// 加载矩阵片段
wmma::load_matrix_sync(a_frag, As[warp_m * WMMA_M], BK);
wmma::load_matrix_sync(b_frag, &Bs[warp_n * WMMA_N][0], BK);

// Tensor Core 计算
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// 存储结果
wmma::store_matrix_sync(&tmp[warp_tile_row][warp_tile_col], c_frag, BN, wmma::mem_row_major);
```

### 6.3 性能对比

#### 综合性能对比 (4096×4096×4096)

| Kernel     | 时间 (ms) | GFLOPS  | 相对 cuBLAS |
| ---------- | --------- | ------- | ----------- |
| cuBLAS     | 31.331    | 4387.38 | 100.00%     |
| Naive      | 280.280   | 490.37  | 11.18%      |
| Shared     | 175.243   | 784.43  | 17.89%      |
| Register   | 120.927   | 1136.77 | 25.93%      |
| Bank       | 122.672   | 1120.40 | 25.55%      |
| DoubleBuf  | 122.048   | 1126.28 | 25.70%      |
| **Tensor FP16** | **47.962** | **2865.68** | **65.37%** |
| **Tensor FP32** | **46.972** | **2926.06** | **66.68%** |

#### 各配置详细性能

| 配置             | 时间 (ms) | GFLOPS  | 相对 cuBLAS |
| --------------- | --------- | ------- | ----------- |
| **256×256 (Tensor FP16)** | 0.100 | 335.91 | 146.98% |
| **256×256 (Tensor FP32)** | 0.157 | 255.43 | 111.77% |
| **1024×1024 (Tensor FP16)** | 2.108 | 1019.53 | 29.77% |
| **1024×1024 (Tensor FP32)** | 1.712 | 1257.35 | 36.72% |
| **4096×4096 (Tensor FP16)** | 47.962 | 2865.68 | 65.37% |
| **4096×4096 (Tensor FP32)** | 46.972 | 2926.06 | 61.37% |
| **4096×1024×8192 (Tensor FP16)** | 26.230 | 2620.13 | 49.75% |
| **4096×1024×8192 (Tensor FP32)** | 26.018 | 2641.60 | 50.16% |
| **8192×512×4096 (Tensor FP16)** | 15.744 | 2183.05 | 46.50% |
| **8192×512×4096 (Tensor FP32)** | 15.230 | 2256.71 | 48.07% |

### 6.4 Tensor Core vs CUDA Core 性能分析

**Tensor Core 显著优于 CUDA Core**：

| 对比 | CUDA Core 最优 (GFLOPS) | Tensor Core (GFLOPS) | 提升倍数 |
| ---- | --------------------- | ------------------- | -------- |
| 256×256 | 456.49 | 335.91 | 0.74x |
| 1024×1024 | 765.89 | 1257.35 | 1.64x |
| 4096×4096 | 1136.77 | 2926.06 | 2.57x |

**关键发现**：
1. Tensor Core 在大矩阵下性能提升显著（2.5x+）
2. 但与 cuBLAS 仍有 35-40% 差距
3. FP32 输出略优于 FP16 输出（更高精度，累加精度更好）

### 6.5 问题分析：为什么 Tensor Core 仍未达到 cuBLAS 水平？

**当前实现的几个关键问题**：

1. **Tile Size 过小**：
   - 当前 `BM=64, BN=64`，只产生 16 个 warp
   - 每个 block 只处理 64×64 输出，并行度不足
   - cuBLAS 使用更大的 tile（如 256×128 或 128×128）

2. **Block Size 配置不当**：
   ```c
   dim3 block(NUM_WARPS * 32);  // 只有 1 个 warp 的线程
   ```
   - 每个 SM 调度效率低，容易产生资源空闲

3. **缺乏 Warp-level Register Blocking**：
   - 当前实现是 Block-level 的，没有充分利用 warp 的并行计算能力
   - WMMA 已经做了 warp-level 优化，但 tile size 限制了整体效果

4. **Shared Memory 布局优化不足**：
   - 没有使用 bank conflict padding
   - 加载效率不如精心优化的 cuBLAS

5. **数据类型转换开销**：
   - 每次 kernel 调用都需要 float→half→float 转换
   - 额外 kernel 启动开销

### 6.6 回答问题

**Q1: Tensor Core 为什么比普通 CUDA core 更适合矩阵乘法？**

> - **专用硬件**：Tensor Core 是专门为矩阵乘法设计的硬件单元，每个时钟周期可执行 128 个 FMA 操作
> - **高吞吐量**：Tensor Core 的矩阵乘加吞吐量远高于 CUDA Core（8-16x）
> - **低精度高效**：Tensor Core 专门针对 FP16/BF16 等低精度设计，能在更短时间内完成更多计算
> - **减少指令数**：一次 `wmma::mma_sync` 完成 16×16×16 次乘加，只需一条指令

**Q2: WMMA 中 fragment 的作用是什么？**

> - **数据打包**：fragment 将多个数据元素打包成一个逻辑单元，便于硬件一次性处理
> - **类型安全**：fragment 模板参数指定数据类型和布局，避免运行时类型错误
> - **内存对齐**：fragment 确保数据按 Tensor Core 要求的方式对齐和布局
> - **累加器管理**：accumulator fragment 管理中间结果的累加

**Q3: Tensor Core 版本是否仍然需要 shared memory staging？**

> - **是的**：Tensor Core 的 `load_matrix_sync` 可以直接从 global memory 加载，但需要满足：
>   - 数据按 16 字节对齐
> - **Shared memory staging 的作用**：
>   - 减少重复加载：同一块数据被多个 warp 复用
>   - 数据格式转换：在 shared memory 中完成列主序/行主序转换
>   - 带宽优化：通过 coalesced 加载提高带宽利用率

**Q4: 为什么 Tensor Core 常与低精度类型结合？**

> - **硬件原生支持**：Tensor Core 原生支持 FP16、BF16、TF32、FP8 等低精度格式
> - **带宽优势**：低精度数据量更小，相同带宽下传输更多数据
> - **功耗效率**：低精度运算功耗更低，性能功耗比更高
> - **精度权衡**：深度学习场景对精度要求相对宽松

**Q5: 性能提升的代价是什么？**

> - **精度损失**：FP16 的动态范围和精度低于 FP32
> - **编程复杂性**：WMMA API 使用比普通 CUDA 更复杂
> - **对齐要求**：数据需要严格对齐，增加了预处理开销
> - **功能限制**：WMMA 只支持特定的矩阵形状和类型组合

---

## 阶段 7: 性能总结与问题分析

### 7.1 各阶段性能总览

| Stage | Kernel        | 4096×4096 GFLOPS | 相对 cuBLAS | 优化要点 |
|-------|--------------|------------------|-------------|----------|
| 0     | cuBLAS       | 4387.38          | 100.00%     | 基准线   |
| 1     | Naive        | 490.37           | 11.18%      | 基准实现 |
| 2     | Shared       | 784.43           | 17.89%      | 数据复用 |
| 3     | Register     | 1136.77          | 25.93%      | 线程分块 |
| 4     | Bank         | 1120.40          | 25.55%      | 冲突消除 |
| 5     | DoubleBuf    | 1126.28          | 25.70%      | 访存流水 |
| 6     | Tensor FP32  | 2926.06          | 66.68%      | 硬件加速 |

### 7.2 性能提升路径

```
cuBLAS (100%)
    │
    │ -5.3x (Naive)
    ▼
Naive (11.18%)
    │
    │ +1.6x (Shared Memory Tiling)
    ▼
Shared Memory (17.89%)
    │
    │ +1.45x (Register Blocking)
    ▼
Register Blocking (25.93%)
    │
    │ ≈ 0x (Bank Conflict / Double Buffering)
    ▼
Bank/DoubleBuf (25.70%)
    │
    │ +2.6x (Tensor Core)
    ▼
Tensor Core (66.68%)
```

### 7.3 关键问题分析

#### 问题 1: 为什么 Bank Conflict 和 Double Buffering 优化无效？

**根本原因**：这两个优化试图解决的不是主要瓶颈

| 优化 | 目标 | 实际情况 |
|------|------|----------|
| Bank Conflict | 减少共享内存访问延迟 | 共享内存延迟（10-20 周期）远低于全局内存（200+ 周期），不是瓶颈 |
| Double Buffering | 隐藏加载延迟 | 当前 kernel 是 bandwidth-bound，不是 latency-bound |

**解决方案**：
- Bank Conflict：需要在 profiling 后确认 shared memory 访问确实是瓶颈时才有意义
- Double Buffering：需要使用 `cp.async` 和 `cuda::pipeline` 真正实现异步加载

#### 问题 2: 为什么 Tensor Core 实现仍未达到 cuBLAS 水平？

| 问题 | 影响 |
|------|------|
| Tile size 太小 (64×64) | 并行度不足，SM 利用率低 |
| Block 只用 1 个 warp | 调度效率低 |
| 缺乏 warp-level 优化 | 未充分利用 Tensor Core 的 warp 并行能力 |
| 数据类型转换开销 | 额外的 kernel 调用和转换时间 |

**cuBLAS 可能使用的优化**：
- 更大的 tile size (如 256×128, 128×128)
- Tensor Core 融合操作（减少数据移动）
- 异步执行和预取
- 更激进的寄存器分配

#### 问题 3: 为什么小矩阵 (256×256) 反而表现更好？

| 矩阵规模 | cuBLAS | Shared Memory | Tensor Core | 说明 |
|----------|--------|--------------|-------------|------|
| 256×256 | 318 GFLOPS | 572 GFLOPS | 335 GFLOPS | 自实现 > cuBLAS |
| 4096×4096 | 4767 GFLOPS | 784 GFLOPS | 2926 GFLOPS | cuBLAS 最佳 |

**原因**：
1. **调度开销占比**：小矩阵下 cuBLAS 的 kernel 启动和资源调度开销占比更大
2. **并行度不足**：自实现 kernel 在小矩阵下仍能充分利用 GPU
3. **cuBLAS 优化方向**：cuBLAS 针对大矩阵优化，小矩阵可能有额外的初始化开销

### 7.4 Roofline 模型分析

#### 各阶段算术强度与性能

| Kernel | 理论算术强度 (FLOP/byte) | 实际 GFLOPS | 瓶颈类型 |
|--------|--------------------------|-------------|----------|
| Naive | ~0.25 | 490 | Memory-bound |
| Shared | ~0.5 | 784 | Memory-bound |
| Register | ~1.0 | 1136 | Memory-bound |
| Tensor Core | ~2.0 (FP16) | 2926 | Memory-bound |

**分析**：
- 所有 kernel 都处于 Memory-bound 区域
- Tensor Core 使用低精度（FP16）提高了有效算术强度
- 实际性能远低于 Roofline 峰值，说明还有其他瓶颈（如对齐、调度开销）

#### Roofline 图示

```
TFLOPS
    ▲
    │                          ★ cuBLAS 峰值
    │                     ★
    │                ★         ● Tensor Core
    │           ●
    │      ●                ● Register/Shared
    │ ●
    │Naive
    └──────────────────────────────────────► 算术强度 (FLOP/byte)
       0.25    0.5    1.0    2.0    4.0
```

### 7.5 Profiling 指标分析

由于缺少详细的 Nsight Compute profiling 数据（roofline_bank.ncu-rep 被删除），根据性能表现推断：

| 指标 | Naive | Shared | Register | Tensor Core |
|------|-------|--------|----------|-------------|
| Global Memory Efficiency | 低 | 中 | 中高 | 高 |
| Shared Memory Efficiency | N/A | 中 | 中 | 中 |
| SM Utilization | 中 | 中高 | 高 | 很高 |
| Occupancy | 高 | 中 | 中低 | 低 |

### 7.6 各优化效果总结

| 优化 | 效果 | 原因 |
|------|------|------|
| Shared Memory Tiling | ✅ 显著 | 减少全局内存重复访问 |
| Register Blocking | ✅ 显著 | 提高线程计算密度 |
| Bank Conflict 消除 | ❌ 无效 | 瓶颈不在共享内存 |
| Double Buffering | ❌ 无效 | 瓶颈不在内存延迟 |
| Tensor Core | ✅ 显著 | 专用硬件加速 |

### 7.7 改进建议

1. **Tensor Core 优化方向**：
   - 增大 tile size 到 128×128 或更大
   - 使用 multi-warp block 充分利用调度器
   - 考虑使用 CUTLASS 库获取更优的实现参考

2. **Double Buffering 优化方向**：
   - 使用 `cp.async` 实现真正的异步加载
   - 增加 tile size 和 BK 值提高计算密度

3. **通用优化方向**：
   - 尝试更大的 block size 提高 occupancy
   - 使用向量化加载（float4）提高带宽利用率
   - 考虑使用 L2 cache 优化

---

## 附录：各阶段性能总结

### 附表 1: 各阶段最优性能汇总

| 阶段 | Kernel      | 最优配置                    | GFLOPS | 相对 cuBLAS |
|------|-------------|---------------------------|--------|-------------|
| 0    | cuBLAS      | -                         | 4387   | 100%        |
| 1    | Naive       | 16×16, 32×8               | 490    | 11.2%       |
| 2    | Shared      | 8×32, BK=32               | 784    | 17.9%       |
| 3    | Register    | 128×128, BK=8, 8×8        | 1137   | 25.9%       |
| 4    | Bank        | 128×128, BK=16, 8×8       | 1120   | 25.6%       |
| 5    | DoubleBuf   | 128×128, BK=16, 8×8       | 1126   | 25.7%       |
| 6    | Tensor Core | 64×64, WMMA, FP32 累加    | 2926   | 66.7%       |

### 附表 2: 各配置性能对比 (4096×4096×4096)

| 配置     | cuBLAS | Naive | Shared | Register | Bank | DoubleBuf | Tensor |
| -------- | ------ | ----- | ------ | -------- | ---- | --------- | ------ |
| 时间(ms) | 31.3   | 280.3 | 175.2  | 120.9    | 122.7| 122.0     | 47.0   |
| GFLOPS   | 4387   | 490   | 784    | 1137     | 1120 | 1126      | 2926   |
| 相对%    | 100%   | 11.2% | 17.9%  | 25.9%    | 25.6%| 25.7%     | 66.7%  |

### 附表 3: 优化收益分析

| 优化步骤              | 收益        | 主要来源                 |
|---------------------|-------------|------------------------|
| Naive → Shared      | +60% GFLOPS | 减少全局内存重复访问    |
| Shared → Register   | +45% GFLOPS | 提高线程计算密度        |
| Register → Bank     | -1.4% GFLOPS| 瓶颈不在共享内存        |
| Register → DoubleBuf| -0.9% GFLOPS| 瓶颈不在内存延迟        |
| DoubleBuf → Tensor  | +160% GFLOPS| 专用硬件加速            |

---

## 实验总结

### 实验完成情况

本次实验完成了从基础 GEMM 到 Tensor Core 优化的完整流程：

1. ✅ **阶段 0-2**：建立实验框架、朴素 CUDA GEMM、Shared Memory 分块
2. ✅ **阶段 3-4**：Register Blocking、Bank Conflict 优化
3. ✅ **阶段 5-6**：Double Buffering、Tensor Core GEMM
4. ⚠️ **阶段 7**：部分完成（缺少详细 Nsight Compute profiling）

### 主要发现

1. **有效优化**：Shared Memory Tiling、Register Blocking、Tensor Core 带来了显著性能提升

2. **无效优化**：Bank Conflict 和 Double Buffering 在当前实现下没有带来性能提升，需要重新审视瓶颈定位

3. **Tensor Core 实现不足**：虽然达到了 cuBLAS 66.7% 的性能，但与工业级实现仍有差距

4. **Memory-bound 特性**：所有 CUDA Core kernel 都处于 memory-bound 区域，性能受限于全局内存带宽

### 未来改进方向

1. 增大 Tensor Core tile size 到 128×128 或更大
2. 使用 `cp.async` 实现真正的异步 double buffering
3. 添加完整的 Nsight Compute profiling 分析
4. 考虑使用 CUTLASS 库作为参考实现

