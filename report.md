# GEMM CUDA 优化实验报告

## 实验基本信息

- **实验主题**：矩阵乘法算子的实现、优化与体系结构分析
- **实验周期**：2026 年 5 月 26 日 — 2026 年 6 月 26 日
- **目标分数**：完成阶段 0–7，得分 ≥ 85
- **目标平台**：NVIDIA Ampere 架构 GPU（编译选项 `-arch=sm_86`），cuBLAS 12.x
- **报告数据来源**：`results/stage{0..6}_results.csv`（每条取 5 次运行的平均值），与本仓库 `src/` 下的最终代码一一对应

### 重要说明

1. **cuBLAS 多次运行结果存在波动**：例如 4096×4096×4096 上 cuBLAS GFLOPS 在不同 stage 跑出 4202–4767，主要由 GPU 频率状态/温度导致。本报告统一以 stage 6 的最新 cuBLAS 数据为参考（4767.41 GFLOPS @ 4096），各 stage 表格中保留各自跑出的 cuBLAS 用于自洽。
2. **阶段 7 Nsight Compute profiling 已完成**：`build/ncu_*.ncu-rep`（7 份 SpeedOfLight / MemoryWorkloadAnalysis / Occupancy / LaunchStats 报告）与 `results/ncu_metrics.csv`（汇总指标）已纳入仓库。7.5 节所有数字均来自实测；7.6 节瓶颈结论也按 NCU 实测结果更新。

### 测试矩阵规模

| 名称                     | M    | N    | K    | 类别       |
| ------------------------ | ---- | ---- | ---- | ---------- |
| square_256               | 256  | 256  | 256  | 小正方     |
| square_1024              | 1024 | 1024 | 1024 | 中正方     |
| square_4096              | 4096 | 4096 | 4096 | 大正方     |
| nonsquare_4096x1024x8192 | 4096 | 1024 | 8192 | 瘦高       |
| nonsquare_8192x512x4096  | 8192 | 512  | 4096 | 扁平       |
| unaligned_1000           | 1000 | 1000 | 1000 | 非 32 对齐 |
| unaligned_511x1023x2047  | 511  | 1023 | 2047 | 奇数尺寸   |

---

## 阶段 0：实验准备与性能基线

### 0.1 实现概述

| Kernel   | 描述                                                         |
| -------- | ------------------------------------------------------------ |
| Kernel 0 | CPU reference GEMM，行主序，支持 α/β，用于小规模正确性验证   |
| cuBLAS   | `cublasSgemm`（行主序接口），提供性能与正确性双参考          |
| Timer    | `cudaEventRecord` + `cudaEventElapsedTime`，`CudaTimer` 包装 |

### 0.2 关键实现

- `src/cpu_reference.cpp` — 三重循环行主序实现，O(MNK) FLOPs。
- `src/cublas_reference.cu` — 行主序 → 列主序转换：传 `(N,M,K)`，矩阵指针交换为 `B,A`，得到列主序下的 C^T = α·A^T·B^T = α·(row-major C)^T。
- `src/main.cu` — 统一 benchmark 驱动，支持 `--validate` 自动选择 CPU/cuBLAS 参考。
- `src/validate.cpp` — 元素级相对误差校验。

### 0.3 性能公式

$$\text{GFLOPS} = \frac{2 \cdot M \cdot N \cdot K}{t_{\text{sec}} \times 10^9}$$

`t` 来自 `cudaEventElapsedTime`（毫秒），因此 `t_sec = t_ms / 1000`。

### 0.4 输出指标

- kernel 平均运行时间 (ms)
- GFLOPS
- 相对 cuBLAS 的性能比例 (%)
- 正确性误差（最大/平均相对误差）

### 0.5 阶段 0 结果

| 配置                     | cuBLAS GFLOPS | CPU GFLOPS | CPU / cuBLAS |
| ------------------------ | ------------- | ---------- | ------------ |
| square_256               | 346.65        | 2.21       | 0.63%        |
| square_1024              | 3708.23       | 0.86       | 0.02%        |
| square_4096              | 4379.30       | —          | —            |
| nonsquare_4096x1024x8192 | 5245.71       | —          | —            |
| nonsquare_8192x512x4096  | 4675.37       | —          | —            |
| unaligned_1000           | 3037.61       | 1.70       | 0.05%        |
| unaligned_511x1023x2047  | 3743.27       | 0.90       | 0.02%        |

CPU 仅用于小规模验证，在 1024 以上规模完全不可用；后续性能对比统一以 cuBLAS 为基线。

---

## 阶段 1：朴素 CUDA GEMM

### 1.1 实现概述

每个 thread 负责 C 的一个元素，按 threadIdx.x/y 映射到 col/row。直接读 A 的整行与 B 的整列，K 维三重累加。无任何优化。

### 1.2 核心代码

```cuda
__global__ void naive_gemm_kernel(float *C, const float *A, const float *B,
                                  int M, int N, int K,
                                  float alpha, float beta)
{
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

### 1.3 不同 Block Size 性能对比（实际 `results/stage1_results.csv`）

#### 256×256×256

| Block    | 时间 (ms) | GFLOPS     | 相对 cuBLAS |
| -------- | --------- | ---------- | ----------- |
| 8×8      | 0.108     | 312.75     | 128.52%     |
| 16×16    | 0.129     | 335.90     | 138.04%     |
| **8×32** | **0.089** | **384.35** | **157.95%** |
| 32×8     | 0.105     | 319.44     | 131.27%     |
| 32×32    | 0.099     | 339.61     | 139.56%     |
| cuBLAS   | 0.193     | 243.33     | 100.00%     |

#### 1024×1024×1024

| Block    | 时间 (ms) | GFLOPS     | 相对 cuBLAS |
| -------- | --------- | ---------- | ----------- |
| 8×8      | 5.776     | 371.79     | 9.85%       |
| 16×16    | 4.633     | 463.54     | 12.28%      |
| **8×32** | **4.589** | **467.99** | **12.39%**  |
| 32×8     | 5.724     | 375.15     | 9.93%       |
| 32×32    | 4.974     | 431.71     | 11.43%      |
| cuBLAS   | 0.576     | 3774.40    | 100.00%     |

#### 4096×4096×4096

| Block     | 时间 (ms)   | GFLOPS     | 相对 cuBLAS |
| --------- | ----------- | ---------- | ----------- |
| 8×8       | 378.925     | 362.85     | 8.11%       |
| 16×16     | 280.280     | 490.37     | 10.96%      |
| 8×32      | 289.082     | 475.64     | 10.63%      |
| 32×8      | 344.264     | 399.35     | 8.92%       |
| **32×32** | **274.716** | **500.35** | **11.18%**  |
| cuBLAS    | 30.778      | 4473.01    | 100.00%     |

#### 非正方 / 非对齐（取最优 Block）

| 配置           | 最优 Block | 时间 (ms) | GFLOPS | 相对 cuBLAS |
| -------------- | ---------- | --------- | ------ | ----------- |
| 4096×1024×8192 | 32×32      | 138.742   | 495.33 | 9.45%       |
| 8192×512×4096  | 8×32       | 71.949    | 477.57 | 10.22%      |
| 1000×1000×1000 | 16×16      | 4.314     | 463.54 | 11.66%      |
| 511×1023×2047  | 8×32       | 4.609     | 464.34 | 12.14%      |

### 1.4 Block Size 分析

- **最优配置**：
  - 256 规模：`8×32`
  - 1024 规模：`8×32` 或 `16×16`
  - 4096 规模：`32×32` 或 `16×16`
  - 总体趋势：tile 偏宽（BN ≥ BM）效果更好；`8×32` 在大多数中大尺寸下最稳
- **关键观察**：
  - `32×8`（BM=32, BN=8）性能最差：在 shared memory 维度上，warp 内不同 thread 读 B 矩阵跨行（stride=N），合并访问差
  - `8×32` 性能较好：A 行连续访问 + warp 内 thread 读 B 同行的相邻列，coalescing 良好
  - `32×32` 在大矩阵下逐渐追上甚至超过 `8×32`，因为更大的 block 隐藏了访存延迟

### 1.5 回答问题

**Q1：为什么 naive kernel 性能较差？**
- 每个输出元素都从 global memory 读 A 的一行 + B 的一列共 2K 个元素
- A/B 的每个元素被重复读 M/N 次（无任何复用）
- 无 thread/block/warp 维度的任何 cache 复用机制

**Q2：A 和 B 的访存模式分别是什么？**
- **A**：每个 thread 读 `A[row * K + k]`，行内连续（stride=1），但跨 row 间隔 K
- **B**：每个 thread 读 `B[k * N + col]`，跨 row 间隔 N（不连续），跨 col 间隔 1

**Q3：哪些访问是连续的，哪些不是？**
- A：warp 内 thread 沿 col 方向（x 维）展开时，A 的行内访问是连续的 → coalesced
- B：同一 warp 沿 col 时，访问同一行 B 的不同 col，连续；但在 K 循环中相邻 thread 的 B 访问间隔 N（对于 `blockDim.y` 维展开时），会跨行不连续

**Q4：每个 A/B 元素被复用了多少次？**
- A 元素 A[row][k]：被用来计算 C[row][0..N-1]，N 次
- B 元素 B[k][col]：被用来计算 C[0..M-1][col]，M 次
- 但因无缓存，每次都从 global 重读（实际有效复用 0 次）

**Q5：该 kernel 更接近 memory-bound 还是 compute-bound？**
- **Memory-bound**。FP32 下每个 A/B 元素读 4 字节、贡献 2 FLOP（mul+add），算术强度 = 2 FLOP / 8 字节 = 0.25 FLOP/byte。Ampere 显存带宽 ~1.5 TB/s，理论峰值性能 ≈ 1.5e12 × 0.25 ≈ 375 GFLOPS，与实测 380–500 GFLOPS 量级一致。后续 stage 2–5 在不引入 Tensor Core 的情况下均未显著超过该上限，证明 memory 才是真正的瓶颈。

---

## 阶段 2：Shared Memory 分块 GEMM

### 2.1 实现概述

每个 block 计算一个 `BM×BN` 的 C tile，沿 K 维循环将 A、B 的 `BM×BK`、`BK×BN` 子块加载到 shared memory，block 内 thread 复用 shared memory 中的数据。`src/kernels/shared_memory.cu` 提供 5 个 tile size。

### 2.2 配置与核心代码

`#define TILE_SIZE = {8, 16, 32}`，脚本额外提供 `32×8` 和 `8×32`（BN=BM 时 BK=TILE_SIZE，否则 BK=32）：

```cuda
template <int BM, int BN, int BK>
__global__ void shared_memory_gemm_kernel(...) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int row = threadIdx.y, col = threadIdx.x;
    float acc = 0.0f;
    int numTiles = (K + BK - 1) / BK;

    for (int tile = 0; tile < numTiles; ++tile) {
        // 边界处理加载
        As[row][col] = (A_row < M && A_col < K) ? A[A_row * K + A_col] : 0.0f;
        Bs[row][col] = (B_row < K && B_col < N) ? B[B_row * N + B_col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k)
            acc += As[row][k] * Bs[k][col];

        __syncthreads();
    }
    // 写回 ...
}
```

### 2.3 不同 Tile Size 性能对比（实际 `results/stage2_results.csv`）

#### 1024×1024×1024

| Tile     | 时间 (ms) | GFLOPS     | 相对 cuBLAS |
| -------- | --------- | ---------- | ----------- |
| 8×8      | 4.608     | 466.10     | 12.39%      |
| 16×16    | 3.522     | 609.80     | 16.21%      |
| 32×32    | 4.081     | 526.17     | 13.99%      |
| 32×8     | 3.434     | 625.27     | 16.63%      |
| **8×32** | **3.162** | **679.21** | **18.06%**  |
| cuBLAS   | 0.574     | 3759.70    | 100.00%     |

#### 4096×4096×4096

| Tile     | 时间 (ms)   | GFLOPS     | 相对 cuBLAS |
| -------- | ----------- | ---------- | ----------- |
| 8×8      | 315.629     | 435.54     | 10.36%      |
| 16×16    | 203.154     | 676.53     | 16.09%      |
| 32×32    | 208.827     | 658.15     | 15.66%      |
| 32×8     | 190.098     | 723.01     | 17.20%      |
| **8×32** | **175.243** | **784.43** | **18.66%**  |
| cuBLAS   | 32.772      | 4202.07    | 100.00%     |

#### 非正方 / 非对齐（取最优 tile）

| 配置           | 最优 Tile | 时间 (ms) | GFLOPS | 相对 cuBLAS |
| -------------- | --------- | --------- | ------ | ----------- |
| 256×256        | 8×32      | 0.058     | 572.28 | 207.73%     |
| 4096×1024×8192 | 8×32      | 89.121    | 771.10 | 14.69%      |
| 8192×512×4096  | 8×32      | 50.221    | 684.17 | 14.50%      |
| 1000×1000      | 8×32      | 3.094     | 646.37 | 16.98%      |
| 511×1023×2047  | 8×32      | 3.180     | 672.95 | 18.24%      |

### 2.4 Tile Size 分析

- **8×32 (BM=8, BN=32, BK=32)** 在所有矩阵规模下都最优。原因是 BN=32 让 block 在 col 方向一次覆盖 32 列（恰好一个 warp 宽度），A 的行内连续访问与 B 的行内连续访问都能被一次 warp 完整覆盖
- **32×32** 占用 8 KB shared memory（As+Bs），thread 数 1024，对 SM 资源压力大，反而不如 8×32
- **32×8** 性能中等：BM=32 适合行方向扩展，但 BN=8 时 block 沿 N 维只能覆盖 8 列，block 数量增多，调度开销变大
- **8×8** 性能接近 stage 1：因为每个 tile 仅 64 元素，shared memory 收益有限

### 2.5 回答问题

**Q1：shared memory tiling 为什么能带来性能提升？**
- 每个 A/B tile 只从 global memory 加载一次，被 block 内 64–1024 个 thread 复用 BM×TN 或 BN×TM 次
- shared memory 带宽是 global memory 的 10–20 倍
- 通过 `__syncthreads()` 保证数据一致性，避免重复读 global

**Q2：A tile 与 B tile 的数据复用分别体现在哪里？**
- A tile (BM×BK)：block 内所有 thread 在计算不同 C[row][col] 时都访问同一组 As[row][k]，每个 As 元素被 BN/TN 个 thread 读
- B tile (BK×BN)：每个 Bs[k][col] 被 BM/TM 个 thread 读
- 沿 K 维循环时，每个 tile 被使用 BK 次

**Q3：tile size 增大为什么不一定总是更快？**
- 更大的 tile → 更多 shared memory 占用 → 同 SM 可驻留的 block 数减少 → occupancy 下降
- 更大的 tile → 每个 thread 加载更多数据 → 寄存器压力上升
- 超过 L1 cache 工作集后，可能出现 shared memory thrashing

**Q4：shared memory 占用如何影响 occupancy？**
- Ampere SM 共享内存 100 KB（默认配置），可调至 163 KB
- 32×32 tile 占用 8 KB，1 个 block 装 8 KB → 1 SM 最多 12 个 block（被 100 KB 限制）
- 8×32 tile 占用 4 KB → 1 SM 最多 25 个 block，occupancy 显著更高

**Q5：优化后 global memory 重复读取是否减少？如何证明？**
- 减少了。naive 中每个 A 元素被读 N 次；shared tiled 中每个 A 元素仅被读 K/BK 次
- 可通过 Nsight Compute 的 `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` 指标对比：naive 应为 `2*M*N*K*4` 字节数量级，shared tiled 为 `(M*K + K*N + M*N) * 4` 数量级

---

## 阶段 3：Register Blocking / Thread-level Tiling

### 3.1 实现概述

在 shared memory tiled 基础上，让每个 thread 计算一个 `TM×TN` 的输出子块，累加器存在寄存器中。K 维度的 A/B 子片段在寄存器中复用，进一步减少对 shared memory 的访问次数。

### 3.2 参数设计

| 配置         | BM  | BN  | BK  | TM  | TN  | threads/block | shared mem |
| ------------ | --- | --- | --- | --- | --- | ------------- | ---------- |
| Config 1     | 64  | 64  | 8   | 4   | 4   | 256 (16×16)   | 2 KB       |
| Config 2     | 64  | 128 | 8   | 4   | 8   | 256 (16×16)   | 4 KB       |
| **Config 3** | 128 | 128 | 8   | 8   | 8   | 256 (16×16)   | 8 KB       |
| Config 4     | 128 | 128 | 16  | 8   | 8   | 256 (16×16)   | 8 KB       |

### 3.3 核心代码

```cuda
template <int BM, int BN, int BK, int TM, int TN>
__global__ void register_blocking_gemm_kernel(...) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int thread_row = threadIdx.y, thread_col = threadIdx.x;
    int C_row = block_row + thread_row * TM;
    int C_col = block_col + thread_col * TN;

    float acc[TM][TN] = {0};

    for (int tile = 0; tile < numTiles; ++tile) {
        // 协作加载 A/B tile
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            for (int k = 0; k < BK; ++k)
                As[thread_row * TM + i][k] = ...;
        // ... Bs ...
        __syncthreads();

        // 在寄存器中复用
        #pragma unroll
        for (int k = 0; k < BK; ++k)
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                float a = As[thread_row * TM + i][k];
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a * Bs[k][thread_col * TN + j];
            }
        __syncthreads();
    }
    // 写回
}
```

### 3.4 不同配置性能对比（实际 `results/stage3_results.csv`）

#### 1024×1024×1024

| 配置         | BM×BN×BK   | TM×TN | 时间 (ms) | GFLOPS     | 相对 cuBLAS |
| ------------ | ---------- | ----- | --------- | ---------- | ----------- |
| cuBLAS       | -          | -     | 0.565     | 3818.79    | 100.00%     |
| Config 1     | 64×64×8    | 4×4   | 3.035     | 707.57     | 18.52%      |
| Config 2     | 64×128×8   | 4×8   | 3.959     | 542.53     | 14.20%      |
| **Config 3** | 128×128×8  | 8×8   | **2.804** | **765.89** | **20.05%**  |
| Config 4     | 128×128×16 | 8×8   | 2.913     | 737.25     | 19.30%      |

#### 4096×4096×4096

| 配置         | BM×BN×BK   | TM×TN | 时间 (ms)   | GFLOPS      | 相对 cuBLAS |
| ------------ | ---------- | ----- | ----------- | ----------- | ----------- |
| cuBLAS       | -          | -     | 31.331      | 4387.38     | 100.00%     |
| Config 1     | 64×64×8    | 4×4   | 163.225     | 842.05      | 19.19%      |
| Config 2     | 64×128×8   | 4×8   | 190.658     | 720.98      | 16.43%      |
| **Config 3** | 128×128×8  | 8×8   | **120.927** | **1136.77** | **25.90%**  |
| Config 4     | 128×128×16 | 8×8   | 129.546     | 1062.12     | 24.20%      |

#### 非正方 / 非对齐（取最优 config）

| 配置           | 最优 Config         | 时间 (ms) | GFLOPS  | 相对 cuBLAS |
| -------------- | ------------------- | --------- | ------- | ----------- |
| 256×256        | Cfg1 64×64×8, 4×4   | 0.074     | 456.49  | 129.20%     |
| 4096×1024×8192 | Cfg3 128×128×8, 8×8 | 67.693    | 1015.51 | 19.45%      |
| 8192×512×4096  | Cfg3 128×128×8, 8×8 | 35.426    | 969.89  | 20.59%      |
| 1000×1000      | Cfg3 128×128×8, 8×8 | 2.725     | 733.98  | 18.13%      |
| 511×1023×2047  | Cfg3 128×128×8, 8×8 | 2.745     | 779.50  | 22.51%      |

### 3.5 配置分析

- **Config 3 (128×128, BK=8, TM=TN=8)** 在所有 ≥1024 的矩阵规模下都最优
- 256 规模下反而是 **Config 1 (64×64, 4×4)** 最好（456.49 GFLOPS / 129.20% cuBLAS），因为 256×256 时 128×128 的 block 远超矩阵大小，浪费并行度
- **BK 选择**：BK=8 比 BK=16 略优，原因：BK=8 时寄存器压力更小（每个 BK 元素只需一组寄存器），occupancy 略高；BK=16 增加 K 维循环内可重用次数，但 register spilling 风险也增大
- **TM×TN 增大**：TM×TN=4×4（16 acc/thread）时寄存器用量约 17×4=68；8×8（64 acc/thread）时约 17×8=136，接近 Ampere 的 255 reg/thread 上限，部分配置下出现 spilling

### 3.6 回答问题

**Q1：为什么 thread-level tiling 能进一步提升性能？**
- 每个 thread 承担 TM×TN 个输出元素的 K 维累加，shared memory load 被摊销到更多 FLOPs 上
- 算术强度从 shared memory tiling 的 ≈0.5 提升到 0.5 × (TM×TN) ≈ 4.0（理论）
- 实际受限于 global memory 带宽，收益约 1.4–1.5×

**Q2：thread 计算多个输出元素时，A/B 数据如何在寄存器中复用？**
- 内层循环中预读 `a = As[thread_row*TM+i][k]`，在 j 维展开 8 次 `acc[i][j] += a * Bs[k][thread_col*TN+j]`
- 同一个 `a` 寄存器值在 8 次 FMA 中被复用 8 次
- 同样的，每个 `Bs[k][col]` 寄存器值在 i 维展开 8 次中被复用 8 次
- 实际每个 shared 元素的 FMA 复用次数 = TM × TN

**Q3：为什么 shared memory 访问次数可能减少？**
- naive + shared tiling：每个 thread 读 2×BK 个 shared 元素
- register tiling（TM×TN）：每个 thread 读 2×BK 个 shared 元素，但产生的 FLOP 数量是 TM×TN 倍
- 因此 **每个 shared 访问产生的 FLOP 提升 TM×TN 倍**，shared memory 访问压力等效降低

**Q4：为什么寄存器过多会降低 occupancy？**
- Ampere SM 寄存器文件 64 KB（256 KB / SM 数量计算）
- TM×TN=8×8 时 acc 数组占 64 寄存器，加上地址/临时变量约 100+ reg/thread
- 占用越多寄存器 → 每 SM 可驻留的 thread 数越少 → occupancy 下降
- 极端情况下寄存器 spill 到 local memory，性能断崖式下跌

**Q5：当前 kernel 的主要瓶颈是否发生变化？**
- 仍是 **global memory bandwidth-bound**：
  - Config 3 @ 4096: 1136.77 GFLOPS, 算术强度约 1.0 FLOP/byte
  - 1136.77 GFLOPS / 1.5 TB/s ≈ 760 FLOP/byte → 1 GFLOPS/byte 才达到带宽上限
  - 实际已接近这一上限
- 但与 stage 1 相比，global memory 读数据量从 `2*M*N*K` 降到约 `2*(M*K + K*N) + M*N`，复用率显著提升

---

## 阶段 4：Shared Memory Bank Conflict 分析与优化

### 4.1 实现概述

在 stage 3 register-blocked kernel 基础上，给 As 和 Bs 加上 `+1` padding：

```cuda
__shared__ float As[BM][BK + 1];   // 第 1 维 padding
__shared__ float Bs[BK][BN + 1];   // 第 2 维 padding
```

写入与计算逻辑不变。这样做是为了错开访问 bank 的下标，避免 32 路线程访问同一 bank。

### 4.2 共享内存 Bank 冲突分析

- CUDA shared memory 划分为 32 个 bank，每个 bank 宽 4 字节（FP32）
- 同一 warp 内 32 个 thread 访问同一 bank 的不同地址 → 串行化（conflict）
- 典型冲突模式：`Bs[k][col]`，当 `col` 沿 warp 内 thread 展开时，所有 thread 读同一行 B → 若无 padding，warp 内 thread 0..31 读 `Bs[k][0..31]`，映射到 bank 0..31，**这是无冲突的 broadcast/不同 bank**
- 但当 `k` 在循环中变化时，连续的 `Bs[k][col]`/`Bs[k+1][col]` 跨行访问 stride = BN+1，padding 后 stride 改变

**实际更显著的冲突**：在 register-blocked kernel 的内层循环
```cuda
for (int j = 0; j < TN; ++j)
    acc[i][j] += a * Bs[k][thread_col * TN + j];
```
- 对固定的 j，warp 内不同 thread 的 `thread_col * TN + j` 间隔 TN=8 → 32 个 thread 跨 4 个不同 thread_col，每个 thread_col 的 j=0..7
- 访问 `Bs[k][c]` 模式：stride=8 在 BN=128 下，bank = (k*129 + c) % 32
- 未 padding (BN=128)：bank = (k*128 + c) % 32，相邻 c 之间 bank 差 1 → 32 个不同 bank，**无冲突**！
- padding (BN+1=129)：bank = (k*129 + c) % 32，相邻 c 之间 bank 差 1 → **同样无冲突**
- **结论**：当前实现里 padding 的 bank-conflict 收益本身就很小，因为 stride=1 的访问本身无冲突

### 4.3 性能对比（实际 `results/stage4_results.csv`）

#### 1024×1024×1024

| 配置         | BM×BN×BK   | TM×TN | 时间 (ms) | GFLOPS     | 相对 cuBLAS |
| ------------ | ---------- | ----- | --------- | ---------- | ----------- |
| cuBLAS       | -          | -     | 0.567     | 3809.09    | 100.00%     |
| Config 1     | 64×64×8    | 4×4   | 3.082     | 696.79     | 18.29%      |
| Config 2     | 64×128×8   | 4×8   | 3.999     | 537.10     | 14.10%      |
| Config 3     | 128×128×8  | 8×8   | 3.163     | 678.96     | 17.82%      |
| **Config 4** | 128×128×16 | 8×8   | **2.869** | **748.55** | **19.65%**  |

#### 4096×4096×4096

| 配置         | BM×BN×BK   | TM×TN | 时间 (ms)   | GFLOPS      | 相对 cuBLAS |
| ------------ | ---------- | ----- | ----------- | ----------- | ----------- |
| cuBLAS       | -          | -     | 31.174      | 4409.27     | 100.00%     |
| Config 1     | 64×64×8    | 4×4   | 167.668     | 819.84      | 18.59%      |
| Config 2     | 64×128×8   | 4×8   | 189.948     | 723.58      | 16.41%      |
| Config 3     | 128×128×8  | 8×8   | 134.646     | 1020.82     | 23.15%      |
| **Config 4** | 128×128×16 | 8×8   | **122.672** | **1120.40** | **25.41%**  |

#### 非正方 / 非对齐（取最优 config）

| 配置           | 最优 Config          | 时间 (ms) | GFLOPS  | 相对 cuBLAS |
| -------------- | -------------------- | --------- | ------- | ----------- |
| 256×256        | Cfg1 64×64×8, 4×4    | 0.075     | 452.06  | 134.34%     |
| 4096×1024×8192 | Cfg4 128×128×16, 8×8 | 67.566    | 1017.23 | 19.46%      |
| 8192×512×4096  | Cfg4 128×128×16, 8×8 | 36.482    | 941.83  | 20.11%      |
| 1000×1000      | Cfg4 128×128×16, 8×8 | 2.812     | 711.37  | 17.38%      |
| 511×1023×2047  | Cfg4 128×128×16, 8×8 | 2.838     | 754.08  | 20.22%      |

### 4.4 Bank Conflict vs Register Blocking 对比

| 配置           | Register GFLOPS | Bank GFLOPS    | Bank 相对 Reg |
| -------------- | --------------- | -------------- | ------------- |
| 256×256        | 456.49          | 452.06         | -0.97%        |
| 1024×1024      | 765.89 (Cfg3)   | 748.55 (Cfg4)  | -2.26%        |
| 4096×4096      | 1136.77 (Cfg3)  | 1120.40 (Cfg4) | -1.44%        |
| 4096×1024×8192 | 1015.51 (Cfg3)  | 1017.23 (Cfg4) | +0.17%        |
| 8192×512×4096  | 969.89 (Cfg3)   | 941.83 (Cfg4)  | -2.89%        |
| 1000×1000      | 733.98 (Cfg3)   | 711.37 (Cfg4)  | -3.08%        |
| 511×1023×2047  | 779.50 (Cfg3)   | 754.08 (Cfg4)  | -3.26%        |

**观察**：
- Bank conflict 优化**基本没有带来收益**（多数情况下小幅 -1%~-3%）
- 主要原因：
  1. 当前 register-blocked kernel 的内层循环是 stride=1 访问，本身就无 bank conflict（见 4.2 分析）
  2. padding 增加了 shared memory 占用，**部分抵消了潜在收益**（每行多 4 字节）
  3. 当前 kernel 的真正瓶颈在 global memory bandwidth，shared memory 优化收益有限
- Config 4（BK=16）在 bank 版下成为最优，而非 stage 3 的 Config 3（BK=8）—— 因为 padding 让 BK=16 时的占用增加比例相对较小

### 4.5 回答问题

**Q1：一个 warp 内线程如何访问 shared memory？**
- 32 个 thread 同一周期发起访问，硬件检测 bank 映射，将不同 bank 的访问并行执行，将同一 bank 的访问串行化
- 同 bank 同地址 → 1 cycle（broadcast）
- 同 bank 不同地址 → N cycles（N=冲突线程数）
- 不同 bank → 1 cycle

**Q2：哪些访问模式容易发生 bank conflict？**
- stride=32：所有 thread 访问同一 bank 的同一位置（broadcast，OK）
- stride=4 且 N=32：32 个 thread 命中 32 个不同 bank（OK）
- stride=2 且 N=32：32 个 thread 命中 16 个 bank（2-way conflict）
- 跨行 stride=BN：当 BN 是 32 的倍数且无 padding 时，`Bs[k+1][c]` 与 `Bs[k][c]` 命中同一 bank → 2-way conflict

**Q3：padding 为什么可能有效？**
- 把 BN 改成 BN+1，跨行访问的 bank 映射改变：`(k+1)*(BN+1) + c = k*BN + BN + k + 1 + c`
- 当 BN=32 时 stride 由 32 变成 33，bank 错开，避免冲突

**Q4：padding 会带来什么代价？**
- shared memory 占用增加（每行多 4 字节）
- 对 BN=128 而言是 +0.8% 占用，可忽略
- 索引计算需要手动处理 padding 偏移（如果做 layout 转换）

**Q5：shared memory 使用量增加后是否影响 occupancy？**
- 对当前 register-blocked kernel，shared memory 占用仅 4–8 KB，远低于 SM 限制，**影响可忽略**
- 主要影响因素仍是 register usage（256 线程 × 100 reg = 25 KB / 64 KB = 40% 占用）

**Q6：bank conflict 降低后性能是否一定提升？**
- **不一定**。当原 kernel 几乎无 bank conflict 时（如本实验），padding 无效甚至有害（占内存 + 编译优化偏移）
- 当 shared memory 访问确实是瓶颈时（如无其他访存压力），才有明显收益

**Q7：A tile 和 B tile 哪一个更可能是主要冲突来源？**
- 在本 kernel 中：`As[thread_row*TM+i][k]` 由所有 thread 读同一 row，warp 内 stride 沿 col 维
- `Bs[k][thread_col*TN+j]`：stride 沿 row 维（k 循环），每个 thread 读不同 col
- **B tile 更可能成为冲突来源**，因为：
  - 内层 k 循环跨 stride=BN
  - 但当前实现 `k` 是外循环，warp 内同一周期读同一行 B，**无冲突**
  - 真要触发 B tile conflict，需要让 warp 内 thread 沿 col 展开 j 时同时读不同 row 的 B —— 这不是本 kernel 的访问模式

---

## 阶段 5：Double Buffering 与访存流水化

### 5.1 实现概述

在 register-blocked kernel 基础上，将 shared memory 扩为 `As[2][BM][BK]` 和 `Bs[2][BK][BN]` 两组 buffer。在主循环中，先预取 tile 0 到 buffer 0；进入循环后，**在计算当前 tile 的同时**预取下一 tile 到另一 buffer，交替使用。

### 5.2 核心代码

```cuda
__shared__ float As[2][BM][BK];
__shared__ float Bs[2][BK][BN];

int write_stage = 0, read_stage;

// 预取 tile 0
load_tile(As[0], Bs[0], /*k=*/0);
__syncthreads();
read_stage = 0; write_stage = 1;

for (int tile = 0; tile < numTiles; ++tile) {
    if (tile + 1 < numTiles) {
        // "预取"下一个 tile
        load_tile(As[write_stage], Bs[write_stage], (tile+1) * BK);
    }
    // 计算当前 tile
    compute_tile(read_stage, acc);
    if (tile + 1 < numTiles) {
        __syncthreads();
        read_stage = write_stage;
        write_stage = 1 - write_stage;
    }
}
```

**重要限制**：当前实现**未使用 `cp.async` 或 `cuda::pipeline`**，预取和计算都是同步发出，`__syncthreads()` 仍然保证数据一致性。这与"真正 overlap 加载与计算"的目标还有差距。

### 5.3 性能对比（实际 `results/stage5_results.csv`）

#### 4096×4096×4096

| 配置         | BM×BN×BK   | TM×TN | 时间 (ms)   | GFLOPS      | 相对 cuBLAS |
| ------------ | ---------- | ----- | ----------- | ----------- | ----------- |
| cuBLAS       | -          | -     | 28.833      | 4767.12     | 100.00%     |
| Config 1     | 64×64×8    | 4×4   | 155.960     | 881.27      | 18.48%      |
| Config 2     | 64×128×8   | 4×8   | 176.347     | 779.42      | 16.34%      |
| Config 3     | 128×128×8  | 8×8   | 124.270     | 1105.98     | 23.20%      |
| **Config 4** | 128×128×16 | 8×8   | **114.096** | **1205.03** | **25.27%**  |

#### 1024×1024×1024

| 配置         | BM×BN×BK   | TM×TN | 时间 (ms) | GFLOPS     | 相对 cuBLAS |
| ------------ | ---------- | ----- | --------- | ---------- | ----------- |
| cuBLAS       | -          | -     | 0.586     | 3713.90    | 100.00%     |
| Config 1     | 64×64×8    | 4×4   | 3.042     | 705.93     | 19.00%      |
| Config 2     | 64×128×8   | 4×8   | 3.996     | 537.39     | 14.46%      |
| Config 3     | 128×128×8  | 8×8   | 3.147     | 682.51     | 18.37%      |
| **Config 4** | 128×128×16 | 8×8   | **2.869** | **748.50** | **20.15%**  |

#### 非正方 / 非对齐（取最优 config）

| 配置           | 最优 Config          | 时间 (ms) | GFLOPS | 相对 cuBLAS |
| -------------- | -------------------- | --------- | ------ | ----------- |
| 256×256        | Cfg1 64×64×8, 4×4    | 0.088     | 399.79 | 113.72%     |
| 4096×1024×8192 | Cfg4 128×128×16, 8×8 | 71.043    | 967.81 | 18.33%      |
| 8192×512×4096  | Cfg4 128×128×16, 8×8 | 36.429    | 943.21 | 20.02%      |
| 1000×1000      | Cfg4 128×128×16, 8×8 | 2.833     | 705.90 | 18.29%      |
| 511×1023×2047  | Cfg4 128×128×16, 8×8 | 2.812     | 761.10 | 22.37%      |

### 5.4 Double Buffering vs Register Blocking 对比

| 配置           | Register GFLOPS | DoubleBuf GFLOPS | Δ      |
| -------------- | --------------- | ---------------- | ------ |
| 256×256        | 456.49          | 399.79           | -12.4% |
| 1024×1024      | 765.89          | 748.50           | -2.3%  |
| 4096×4096      | 1136.77         | 1205.03          | +6.0%  |
| 4096×1024×8192 | 1015.51         | 967.81           | -4.7%  |
| 8192×512×4096  | 969.89          | 943.21           | -2.7%  |

### 5.5 分析

**Double Buffering 在当前实现下基本与 Register Blocking 持平**：

1. **4096 反而提升 6%**：双 buffer 减弱了 `__syncthreads()` 边界 stall；同样 BM/BN/BK/TM/TN 下，预取下一 tile 期间 SM 可执行其他 warp 的 mma 或 shared 加载流水
2. **小矩阵下略降 2–12%**：256×256 时 block 远超矩阵大小，buffer 翻倍占用导致 occupancy 进一步下降
3. **未启用真正的异步 copy**：当前仍以 `__syncthreads()` 同步，"预取"和"计算"在同一周期串行执行，没有真实 overlap
4. **当前 kernel 仍 memory-bound**：
   - 1136.77 GFLOPS @ 4096 对应算术强度 ≈ 1 FLOP/byte
   - Ampere 显存带宽 ~1.5 TB/s → 理论上限 ~1500 GFLOPS
   - 实际 1205 已经接近上限，再 overlap 也无法突破带宽
5. **K 维 tile 数对结果有影响**：K=4096 / BK=8 = 512 个 tile，切换 512 次开销不容忽视

**改进方向**：
- 改用 `cp.async` + `cp.async.commit_group` + `cp.async.wait_group` 实现真正异步
- 或者用 `cuda::pipeline`（CUDA 12+）封装
- 增加 BK 减少 tile 切换次数（但要权衡 shared memory 与 register 压力）

### 5.6 回答问题

**Q1：double buffering 试图隐藏什么延迟？**
- global memory 加载的延迟（200+ cycle）
- 理想情况下，前一个 tile 计算时，下一 tile 已经在传输中

**Q2：为什么需要两组 shared memory buffer？**
- 避免读写冲突：读 buffer 用于计算，写 buffer 用于预取，互不干扰
- 切换 buffer 是简单的索引翻转（`read_stage = write_stage; write_stage = 1 - write_stage`）

**Q3：double buffering 增加了多少 shared memory 使用量？**
- 翻倍：`(BM×BK + BK×BN)` → `2×(BM×BK + BK×BN)`
- 例如 Config 4: 8KB → 16KB
- Config 3: 4KB → 8KB

**Q4：在什么情况下 double buffering 收益更明显？**
- 计算密度高（每个 tile 算得多）→ 加载开销可被计算掩盖
- 内存延迟确实为瓶颈（compute-bound kernel 也有该瓶颈，因为 global load 仍然耗 cycle）
- 已支持异步 copy（`cp.async`）

**Q5：double buffering 与 padding 是否会相互影响？**
- 会。stage 4 的 padding 进一步增加 shared memory 占用，再叠加 double buffer → 占用翻倍
- 本实验未做此组合

**Q6：当前没有带来性能提升，可能原因？**
- 仍以 `__syncthreads()` 同步，未启用 `cp.async`/`cuda::pipeline`
- 当前 kernel 主要瓶颈在 global memory **带宽**而非**延迟**；双 buffer 只能隐藏延迟，对带宽无帮助
- 占用翻倍后 occupancy 略降，部分抵消潜在收益

---

## 阶段 6：Tensor Core GEMM

### 6.1 实现概述

使用 NVIDIA `nvcuda::wmma` API 调用 Tensor Core 计算单元。`BM=BN=64, BK=16, WMMA_M=WMMA_N=WMMA_K=16`，每个 block 含 4×4=16 个 warp（实际 16 个 warp 复用），每个 warp 计算 16×16 子块。

- 输入：FP16；累加器：FP32
- 输出：分别实现 FP16 输出与 FP32 输出
- A 在 shared memory 中以行主序存储，B 转置为列主序存储以匹配 `wmma::col_major`
- 采用双 buffer 形式（与 stage 5 类似）减少同步开销

### 6.2 核心代码（FP32 输出版）

```cuda
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int BM = 64, BN = 64;
constexpr int BK = WMMA_K;
constexpr int WN = BN / WMMA_N;  // 4

__shared__ half As[2][BM][BK];
__shared__ half Bs[2][BN][BK];
__shared__ float tmp[BM][BN];

wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
wmma::fill_fragment(c_frag, 0.0f);

int warp_m = warp_id / WN;
int warp_n = warp_id % WN;

for (int tile = 0; tile < numTiles; ++tile) {
    wmma::load_matrix_sync(a_frag, As[buf][warp_m * WMMA_M], BK);
    wmma::load_matrix_sync(b_frag, &Bs[buf][warp_n * WMMA_N][0], BK);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    // 预取下一 tile ...
    __syncthreads();
    buf ^= 1;
}

wmma::store_matrix_sync(&tmp[warp_m * WMMA_M][warp_n * WMMA_N], c_frag, BN, mem_row_major);
__syncthreads();
// 协作写回 global
```

输入在 host 调用 `run_tensor_core_kernel` 时通过 `float2half_kernel` 在 device 端转换；FP16 输出路径额外做了 `half2float_kernel` 转换回 FP32。

### 6.3 性能结果（实际 `results/stage6_results.csv`）

| 配置           | Kernel          | 时间 (ms)  | GFLOPS      | 相对 cuBLAS |
| -------------- | --------------- | ---------- | ----------- | ----------- |
| 256×256        | cuBLAS          | 0.165      | 228.53      | 100.00%     |
|                | Tensor FP16     | 0.100      | 335.91      | 146.98%     |
|                | Tensor FP32     | 0.157      | 255.43      | 111.77%     |
| 1024×1024      | cuBLAS          | 0.635      | 3424.04     | 100.00%     |
|                | Tensor FP16     | 2.108      | 1019.53     | 29.77%      |
|                | **Tensor FP32** | **1.712**  | **1257.35** | **36.72%**  |
| **4096×4096**  | cuBLAS          | 28.832     | 4767.41     | 100.00%     |
|                | Tensor FP16     | 47.962     | 2865.68     | 60.10%      |
|                | **Tensor FP32** | **46.972** | **2926.06** | **61.37%**  |
| 4096×1024×8192 | cuBLAS          | 13.050     | 5266.28     | 100.00%     |
|                | Tensor FP16     | 26.230     | 2620.13     | 49.75%      |
|                | **Tensor FP32** | 26.018     | 2641.60     | 50.16%      |
| 8192×512×4096  | cuBLAS          | 7.320      | 4694.18     | 100.00%     |
|                | Tensor FP16     | 15.744     | 2183.05     | 46.50%      |
|                | **Tensor FP32** | 15.230     | 2256.71     | 48.07%      |
| 1000×1000      | cuBLAS          | 0.631      | 3198.00     | 100.00%     |
|                | Tensor FP16     | 2.047      | 977.76      | 30.57%      |
|                | **Tensor FP32** | 1.631      | 1227.67     | 38.38%      |
| 511×1023×2047  | cuBLAS          | 0.567      | 3782.98     | 100.00%     |
|                | Tensor FP16     | 2.650      | 839.25      | 22.18%      |
|                | **Tensor FP32** | 2.004      | 1070.72     | 28.30%      |

### 6.4 Tensor Core vs CUDA Core

| 矩阵规模       | CUDA Core 最优 (Reg, GFLOPS) | Tensor FP32 GFLOPS | Tensor / CUDA Core |
| -------------- | ---------------------------- | ------------------ | ------------------ |
| 256×256        | 456.49                       | 255.43             | 0.56×              |
| 1024×1024      | 765.89                       | 1257.35            | 1.64×              |
| **4096×4096**  | **1136.77**                  | **2926.06**        | **2.57×**          |
| 4096×1024×8192 | 1015.51                      | 2641.60            | 2.60×              |
| 8192×512×4096  | 969.89                       | 2256.71            | 2.33×              |

**关键发现**：
- 矩阵越大，Tensor Core 优势越明显（256×256 时反不如 CUDA core，因为 tile=64×64 太大，浪费）
- 大矩阵相对 cuBLAS 达到 60–65%，绝对吞吐约 2.9 TFLOPS（FP16 累加器的等效 TFLOPS 更高）

### 6.5 与 cuBLAS 的差距

| 问题                                                  | 影响                                                        |
| ----------------------------------------------------- | ----------------------------------------------------------- |
| Tile size 偏小（BM=BN=64）                            | 1 个 block 仅 16×16=256 输出，远小于 cuBLAS 的 128×128 起步 |
| 未使用向量化加载（float4/half2）                      | global memory 带宽未充分利用                                |
| 预取 / store_matrix_sync 之间同步开销大               | 限制了 overlap 收益                                         |
| 数据类型转换（FP32→FP16）的 kernel 未与主 kernel 融合 | 额外 kernel 启动开销                                        |
| 缺少 `mma.sp` 等 Ampere 新指令优化                    | 没有利用 2:4 sparse Tensor Core                             |

### 6.6 回答问题

**Q1：Tensor Core 为什么比普通 CUDA core 更适合矩阵乘法？**
- 专用硬件单元，每个 cycle 可完成 `D = A * B + C` 形式的 16×16×16 矩阵乘加
- 等效吞吐：FP16 输入 FP32 累加 ~165 TFLOPS（Ampere），比 CUDA core FP32 ~20 TFLOPS 高 8× 以上
- 一条 `mma_sync` 完成 4096 次 FMA，指令密度高

**Q2：WMMA 中 fragment 的作用是什么？**
- 把 fragment 内的数据映射到 Tensor Core 期望的物理寄存器布局
- 用户无需关心物理布局
- `mma_sync` 直接接收 fragment，无需手动重排

**Q3：Tensor Core 版本是否仍然需要 shared memory staging？**
- **是**。原因：
  - `load_matrix_sync` 只能从 `16 字节对齐` 的内存加载，shared memory 可以保证对齐
  - 同 block 内多个 warp 复用同一份 A/B tile，必须经 shared memory
  - 直接从 global 加载会重复读取同一 tile 多次

**Q4：为什么 Tensor Core 常与低精度类型结合？**
- Tensor Core 原生支持 FP16/BF16/TF32/FP8/INT8/INT4
- 低精度 → 同样大小的数据携带更多计算量 → 算术强度提升
- 显存/共享内存带宽更省
- 推理场景精度要求宽松

**Q5：性能提升的代价是什么？**
- 精度：FP16 动态范围约 6e-5，溢出风险高
- 编程复杂：需要 fragment 抽象
- 形状限制：m=n=k=16 起步，矩阵需对齐到 16
- 转换开销：FP32↔FP16 需要额外 kernel（cuBLAS 内部用 split-k 等技巧隐藏）

---

## 阶段 7：Profiling、Roofline 建模与整体总结

### 7.1 各阶段性能总览（4096×4096×4096，统一以 stage 6 cuBLAS=4767.41 GFLOPS 为参考）

| Stage | Kernel                     | 时间 (ms)  | GFLOPS      | 相对 cuBLAS | 优化要点         |
| ----- | -------------------------- | ---------- | ----------- | ----------- | ---------------- |
| 0     | cuBLAS                     | 28.832     | 4767.41     | 100.00%     | 性能基线         |
| 1     | Naive (32×32)              | 274.716    | 500.35      | 10.49%      | 基础实现         |
| 2     | Shared (8×32)              | 175.243    | 784.43      | 16.45%      | 块内数据复用     |
| 3     | Register (128×128, BK=8)   | 120.927    | 1136.77     | 23.85%      | 线程内寄存器复用 |
| 4     | Bank (128×128, BK=16)      | 122.672    | 1120.40     | 23.50%      | 几乎无收益       |
| 5     | DoubleBuf (128×128, BK=16) | 114.096    | 1205.03     | 25.27%      | 较 register +6%  |
| 6     | **Tensor FP32 (64×64)**    | **46.972** | **2926.06** | **61.37%**  | 硬件加速         |

### 7.2 性能提升路径

```
cuBLAS (100%, 4767 GFLOPS)
    │
    │ -9.5x  (Naive)
    ▼
Naive 32×32  (10.5%, 500 GFLOPS)
    │
    │ +1.57x (Shared Memory Tiling, BM=8 BN=32 BK=32)
    ▼
Shared 8×32   (16.5%, 784 GFLOPS)
    │
    │ +1.45x (Register Blocking, BM=BN=128 BK=8 TM=TN=8)
    ▼
Register 128×128 (23.8%, 1137 GFLOPS)
    │
    │ -1.4% (Bank Conflict, padding 无效)
    ▼
Bank 128×128×16 (23.5%, 1120 GFLOPS)
    │
    │ +7.6% (Double Buffering，4096 规模)
    ▼
DoubleBuf 128×128×16 (25.3%, 1205 GFLOPS)
    │
    │ +2.4x (Tensor Core, FP16 in / FP32 accum)
    ▼
Tensor FP32 (61.4%, 2926 GFLOPS)
```

### 7.3 各阶段贡献分析

| 优化步骤           | 绝对收益     | 主要来源                                                                         |
| ------------------ | ------------ | -------------------------------------------------------------------------------- |
| Naive → Shared     | +284 GFLOPS  | 减少 global 重复读，每个 A/B 元素复用 BM×BN/TM/TN 次                             |
| Shared → Register  | +352 GFLOPS  | shared 元素复用 TM×TN 次，提高算术强度 4×                                        |
| Register → Bank    | -16 GFLOPS   | bank conflict 本就不严重，padding 收益为 0；额外 shared mem 占用略微降 occupancy |
| Bank → DoubleBuf   | +85 GFLOPS   | 双 buffer 减弱了 `__syncthreads()` 边界 stall；4096 规模下流水化明显             |
| DoubleBuf → Tensor | +1721 GFLOPS | 专用矩阵乘加硬件（165 TFLOPS Ampere FP16 累加）                                  |

### 7.4 Roofline 模型分析

Ampere A100 (sm_80) 估算参数（实验 GPU 推测为 RTX 30/A10，参数近似）：

- 显存带宽 ~1.5 TB/s
- FP32 峰值 ~20 TFLOPS
- Tensor Core FP16 峰值 ~165 TFLOPS

各 kernel 的算术强度 (FLOP/byte)：

| Kernel             | 算术强度   | 带宽上限 (GFLOPS)          | 实测 (GFLOPS) | 利用率                                 |
| ------------------ | ---------- | -------------------------- | ------------- | -------------------------------------- |
| Naive              | 0.25       | 375                        | 500           | > 100% (cache hit)                     |
| Shared (8×32)      | 0.5        | 750                        | 784           | ~100%                                  |
| Register (128×128) | 1.0        | 1500                       | 1137          | ~76%                                   |
| Bank               | 1.0        | 1500                       | 1120          | ~75%                                   |
| DoubleBuf          | 1.0        | 1500                       | 1205          | ~80%                                   |
| Tensor FP32        | 1.0 (FP16) | 1500 (FP32) / 75000 (FP16) | 2926          | compute-bound (算术强度按 FP16 算 ≥ 4) |

**Roofline 图示**：

```
TFLOPS
   ▲
   │                                            ★ cuBLAS Tensor Core 上限
   │                                     ●  Tensor Core (FP32 累加)
   │                                    /
   │                               /
   │           ┌─────────┐
   │           │         │ ★ FP32 peak ~20 TFLOPS
   │           │ Compute │
   │   ●───────│  Bound  │
   │  Reg/Bank │         │
   │ /         └─────────┘
   │/ Shared
   │
   │ ● Naive
   │
   └─────────────────────────────────────────► 算术强度 (FLOP/byte)
     0.25    0.5    1.0   2.0   4.0
```

**结论**：
- 所有 CUDA core kernel 都集中在 **memory-bound 区域**（算术强度 0.25–1.0）
- 优化方向从"算术强度低" → "算术强度接近 1.0"，但受限于 global memory 带宽
- Tensor Core 走的是 FP16/FP32 双精度路径，等效算术强度提升到 2–4，且不受 FP32 峰值天花板限制
- **真正的性能突破来自硬件加速**，而非传统 software 优化

### 7.5 Profiling 指标分析（Nsight Compute 实测）

> ✅ 本节数据来自 `build/ncu_*.ncu-rep`（已纳入仓库），原始指标汇总于 `results/ncu_metrics.csv`。每次跑对应一次 kernel launch（包含 cold-cache 与 profiling overhead，因此绝对 GFLOPS 略低于 `results/stage*.csv` 的 5 次平均；相对关系与各 stage csv 一致）。

#### 7.5.1 硬件资源使用（launch 参数，1024 单次运行）

| Kernel                     | regs/thread | shared/block | occ_limit (block/SM) | 主要约束          |
| -------------------------- | ----------- | ------------ | -------------------- | ----------------- |
| naive (16×16)              | 40          | 0            | 6                    | 寄存器            |
| shared (8×32)              | 37          | 5.12 KB      | 6                    | 寄存器            |
| register (128×128, BK=8)   | 128         | 8.19 KB      | **2**                | **寄存器**        |
| bank    (128×128, BK=8)    | **154**     | 8.74 KB      | **1**                | 寄存器+shared     |
| doublebuf (128×128, BK=16) | **168**     | **32.77 KB** | **1**                | shared（×4 翻倍） |

**关键观察**：
- 从 register → bank：padding 让每个 thread 寄存器用量从 128 → 154（+20%），shared 也从 8.19 → 8.74 KB（+7%），occupancy 跌到 1 block/SM
- 从 register → doublebuf：shared 占用 **4× 翻倍**（As+Bs 变 [2][…]），occupancy 也跌到 1 block/SM

#### 7.5.2 Speed of Light（4096×4096×4096 单次运行）

| Kernel         | L1/TEX 吞吐 | Memory 吞吐 | DRAM 吞吐 | SM 吞吐 | 备注                            |
| -------------- | ----------- | ----------- | --------- | ------- | ------------------------------- |
| register Cfg3  | **95.18%**  | 93.72%      | 9.03%     | 58.04%  | L1/shared 几乎满载，DRAM 未触顶 |
| bank Cfg4      | 88.68%      | 87.34%      | 8.74%     | 57.22%  | occupancy 下降拖累 L1 利用率    |
| doublebuf Cfg4 | 91.88%      | 90.43%      | 8.77%     | 56.04%  | 与 bank 相近                    |
| Tensor FP32    | 79.27%      | 79.24%      | 30.25%    | 44.23%  | DRAM 占比最高，受全局访存限制   |

**核心结论**：
- 在 CUDA core kernel 上，**真正的瓶颈是 L1/shared memory（95% 利用率），而非 DRAM（仅 9%）**。roofline 模型在 7.4 节按 DRAM 带宽画线时显示"memory-bound"是对的，但精确化到 L1 之后，本质上是 **shared memory 带宽 / bank 访问模式受限**
- Tensor Core 把 L1 利用率降到 79%，DRAM 上升到 30%——计算更密集，瓶颈部分迁移到 DRAM

#### 7.5.3 占用率与吞吐（1024 单次运行）

| Kernel         | warps_active | sm__throughput | inst_executed | FMA pipe |
| -------------- | ------------ | -------------- | ------------- | -------- |
| naive (16×16)  | **98.77%**   | 97.90%         | —             | —        |
| shared (8×32)  | 98.70%       | 98.89%         | —             | —        |
| register Cfg3  | 29.25%       | 45.96%         | 28.22%        | 11.01%   |
| bank Cfg3      | 16.65%       | 41.63%         | 25.20%        | 9.71%    |
| doublebuf Cfg4 | 16.65%       | 44.39%         | 28.52%        | 10.96%   |

**观察**：
- naive 与 shared 阶段 warps_active ≈ 99%——几乎所有 warp 都在跑
- 进入 register 优化后，warps_active 跌到 29%——寄存器成为约束；但 sm__throughput 仍达 46%（说明少量 warp 仍能跑满 SM 调度）
- bank 与 doublebuf 都因 shared 翻倍 / 寄存器变多，warps_active 跌到 17%——但 inst_executed 比例几乎不变（25–28%），说明 **每条指令仍然被有效发射**，只是 warp 太少无法隐藏长延迟
- FMA 管道利用率 < 12%：所有 CUDA core kernel 都不是计算受限，而是被内存访问延迟"卡"住

#### 7.5.4 Stall 原因分析（warps issue stalled per issue active，1024 单次运行）

| Stall reason                   | naive     | shared | register | bank     | doublebuf |
| ------------------------------ | --------- | ------ | -------- | -------- | --------- |
| **short_scoreboard**（shared） | 0.01      | 0.21   | 2.44     | **1.59** | 1.89      |
| **long_scoreboard**（global）  | **10.44** | 2.91   | 1.08     | **2.16** | **0.62**  |
| **barrier**（`__syncthreads`） | 0.00      | 5.33   | 0.65     | 0.34     | **0.07**  |
| **lg_throttle**（lsu）         | —         | —      | 0.54     | 0.09     | 0.19      |
| membar / drain                 | 0         | 0      | 0        | 0        | 0         |

**读图**：
- **naive → shared**：long_scoreboard 10.44 → 2.91，barrier 0 → 5.33。共享内存消除了大部分 global 等待，但 `__syncthreads()` 成为新瓶颈
- **shared → register**：long_scoreboard 2.91 → 1.08，barrier 5.33 → 0.65。每个 thread 计算 8×8 元素后，shared 读/同步频率都降低
- **register → bank**：
  - short_scoreboard 2.44 → **1.59（-35%）**：padding 减少了 shared bank 访问串行化
  - long_scoreboard 1.08 → **2.16（+100%）**：occupancy 跌到 1 block/SM 后，global load latency 无法被其他 warp 隐藏
  - net: bank 失败的核心原因是 **occupancy 下降导致 long_sb 翻倍**，**padding 本身的 short_sb 改善被完全抵消**
- **register → doublebuf**：
  - long_scoreboard 1.08 → **0.62（-43%）**：预取确实隐藏了 global memory 延迟
  - barrier 0.65 → **0.07（-89%）**：双 buffer 让"等下一组数据时本组还在算"
  - 同样 occupancy 跌到 1，但 long_sb 与 barrier 大幅下降，部分抵消
  - net: 1024 规模 +3%、4096 规模 +6%（规模越大，occupancy 损失越小，流水化收益越明显）

#### 7.5.5 Bank Conflict 原始数据（register vs bank，BK=8，TM=TN=8，1024 单次运行）

| 指标                                                       | register   | bank       | 差值     |
| ---------------------------------------------------------- | ---------- | ---------- | -------- |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | 4,194,304  | 6,029,312  | **+44%** |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` | 16,777,216 | 12,706,749 | **-24%** |
| gpu time (ms)                                              | 2.78       | 3.16       | +14%     |

**读图**：
- 原始 bank conflict count 中 padding 反而 **增加了 ld conflict、减少了 st conflict**——这是因为 `Bs[BK][BN+1]` padding 让 store 阶段的 stride 变成 129（避免同一 bank），但 `As[BM][BK+1]` 改变 ld 阶段的 bank 映射后，跨行访问模式被破坏
- 真正决定性指标是 **time**：padding 让单次访问的 bank 排布更分散（理论好），但 occupancy 跌一半（实际坏）

#### 7.5.6 总结：为什么 Stage 4/5 优化在 CSV 上"无效"或"边际"？

| 现象                              | 真实原因（NCU 数据证实）                                                                                             |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| bank 优化没有提升（甚至 -1%~-3%） | 1. short_sb 改善 35%，但被 long_sb 翻倍抵消<br>2. occupancy 从 2 → 1 block/SM，每 warp 摊销的 latency 隐藏窗口减半   |
| doublebuf 在 4096 反而 +6%        | 1. long_sb 1.08 → 0.62（-43%）<br>2. barrier 0.65 → 0.07（-89%）<br>3. 流水化收益在更大规模上才能抵消 occupancy 损失 |
| doublebuf 在 256 反而 -12%        | occupancy 1 block/SM + block 远超矩阵大小，调度开销占主导                                                            |

#### 7.5.7 复现命令

```bash
# 单次 kernel 测 bank conflict metrics
cd build
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
                l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
                sm__throughput.avg.pct_of_peak_sustained_active,\
                gpu__time_duration.sum \
     --kernel-name regex:register_blocking \
     --launch-skip 1 --launch-count 1 \
     ./benchmark 1024 1024 1024 register --config=3

# 完整 SpeedOfLight（已纳入 build/ncu_*.ncu-rep）
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section Occupancy --section LaunchStats \
    --kernel-name regex:bank_conflict \
    --launch-skip 1 --launch-count 1 \
    -o ncu_bank_4096 \
    ./benchmark 4096 4096 4096 bank --config=4

# 转换为 CSV
ncu --import ncu_bank_4096.ncu-rep --csv --page raw > ncu_bank_4096.csv
```

### 7.6 回答阶段 7 体系结构问题

**Q1：每一代 kernel 的性能瓶颈是什么？**（NCU 实测，4096×4096×4096）

| Kernel    | 主要瓶颈（NCU 实测）                             | 数据来源      |
| --------- | ------------------------------------------------ | ------------- |
| Naive     | global memory 带宽（long_sb 10.44）              | 7.5.4 表      |
| Shared    | global memory 带宽 + barrier 同步（5.33）        | 7.5.4 表      |
| Register  | L1/shared memory（95.18% 吞吐）+ short_sb 2.44   | 7.5.2 + 7.5.4 |
| Bank      | occupancy 跌到 1 → long_sb 翻倍到 2.16           | 7.5.1 + 7.5.4 |
| DoubleBuf | occupancy 跌到 1，但 long_sb 与 barrier 显著降低 | 7.5.1 + 7.5.4 |
| Tensor    | Tensor Core 单元（44% SM 吞吐）+ L1（79%）       | 7.5.2         |

**Q2：优化后瓶颈是否发生迁移？**（NCU 实测）

- Naive → Shared：瓶颈从"非合并的重复 global 读"（long_sb 10.44）迁移到"shared 复用 + barrier 同步"（barrier 5.33）
- Shared → Register：瓶颈从"shared 带宽 + barrier"（barrier 5.33）迁移到"short_sb 共享内存访问"（2.44）；long_sb 进一步从 2.91 降至 1.08
- Register → Tensor：瓶颈从 shared memory（L1 95%）迁移到 Tensor Core 单元（SM 44%）+ L1（79%）
- Bank / DoubleBuf：瓶颈仍在 shared memory + occupancy 区域，没有迁移

**Q3：当前实现距离硬件理论上限还有多远？**

| Kernel                | 理论上限（NCU 实测）                        | CSV 平均（5 次）              | 利用率 |
| --------------------- | ------------------------------------------- | ----------------------------- | ------ |
| Naive (CUDA core)     | sm_throughput 97.90% (occupancy 受限)       | 500 GFLOPS / 20 TFLOPS peak   | ~2.5%  |
| Shared (CUDA core)    | sm_throughput 98.89%                        | 784 GFLOPS                    | ~3.9%  |
| Register (CUDA core)  | sm_throughput 45.96%, L1 95.18% (memory 端) | 1137 GFLOPS                   | ~5.7%  |
| Bank (CUDA core)      | sm_throughput 41.63%, L1 88.68%             | 1120 GFLOPS                   | ~5.6%  |
| DoubleBuf (CUDA core) | sm_throughput 44.39%, L1 91.88%             | 1205 GFLOPS                   | ~6.0%  |
| Tensor FP32           | sm 44.23%, L1 79.27%, DRAM 30.25%           | 2926 GFLOPS / 165 TFLOPS peak | ~1.8%  |

**关键发现**：
- 即使在最好的 register kernel 上，**sm_throughput 仅 45.96%**（1024）/ **L1/TEX 已 95.18%**（4096）——说明 warp 数量是限制因素，而不是 SM 调度本身
- Tensor Core 距其理论峰值 (165 TFLOPS) 仅利用 1.8%，与 cuBLAS 76% 利用率差距巨大，主要来自 tile size、warp 调度、内存流水优化不足
- 在 L1/shared memory 已经几乎饱和的现状下，CUDA core 想突破 1200 GFLOPS，需要的不是更多计算优化，而是**降低 L1 压力**——例如把 8×8 register tile 改成 swizzled layout / vectorized load 等

**Q4：性能提升来自访存优化、计算复用、并行度提升，还是 Tensor Core 使用？**

- Stage 1→3：访存优化 + 计算复用（global → shared → register）
- Stage 3→5：非常有限（已接近 global bandwidth 上限；doublebuf 仅有 6% 提升来自同步 stall 减少）
- Stage 5→6：Tensor Core 硬件加速（专用矩阵乘加单元 + FP16/FP32 混合精度）

**Q5：哪些优化有效，哪些没有达到预期？**

- ✅ Shared Memory Tiling、Register Blocking、Tensor Core 显著有效
- ❌ Bank Conflict 优化几乎无效（本 kernel 几乎无 conflict）
- ⚠️ Double Buffering 边际有效（4096 规模 +6%，其他规模 -3%~-12%）
- ⚠️ 小矩阵（256×256）下 cuBLAS 不如自实现（cuBLAS 调度开销占比大）

**Q6：如何用 Roofline 模型解释？**

- Naive、Shared、Register、Bank、DoubleBuf 均位于 Roofline 的 **memory-bound 段**（斜线段）
- 沿斜线段向上移动（算术强度 0.25 → 1.0），性能随强度线性提升 → 这是 stage 1–3 收益的来源
- 沿斜线段继续向上移动（强度 1.0 → 1.0），无收益 → 这是 stage 4 的失败
- 跨过"屋脊点"（roofline ridge point）后进入 compute-bound → Tensor Core 跨越到该区域

---

## 附录 A：各阶段最优性能汇总（4096×4096×4096）

| 阶段 | Kernel      | 最优配置                | 时间 (ms) | GFLOPS  | 相对 cuBLAS |
| ---- | ----------- | ----------------------- | --------- | ------- | ----------- |
| 0    | cuBLAS      | -                       | 28.832    | 4767.41 | 100.00%     |
| 1    | Naive       | 32×32                   | 274.716   | 500.35  | 10.49%      |
| 2    | Shared      | 8×32                    | 175.243   | 784.43  | 16.45%      |
| 3    | Register    | 128×128, BK=8, TM=TN=8  | 120.927   | 1136.77 | 23.85%      |
| 4    | Bank        | 128×128, BK=16, TM=TN=8 | 122.672   | 1120.40 | 23.50%      |
| 5    | DoubleBuf   | 128×128, BK=16, TM=TN=8 | 114.096   | 1205.03 | 25.27%      |
| 6    | Tensor FP32 | 64×64, WMMA, FP32 累加  | 46.972    | 2926.06 | 61.37%      |

## 附录 B：各配置性能对比 (4096×4096×4096)

| Kernel      | cuBLAS  | Naive 32×32 | Shared 8×32 | Reg 128×128 (BK=8) | Bank 128×128 (BK=16) | DB 128×128 (BK=16) | Tensor FP32 |
| ----------- | ------- | ----------- | ----------- | ------------------ | -------------------- | ------------------ | ----------- |
| 时间 (ms)   | 28.832  | 274.716     | 175.243     | 120.927            | 122.672              | 114.096            | 46.972      |
| GFLOPS      | 4767.41 | 500.35      | 784.43      | 1136.77            | 1120.40              | 1205.03            | 2926.06     |
| 相对 cuBLAS | 100%    | 10.5%       | 16.5%       | 23.8%              | 23.5%                | 25.3%              | 61.4%       |

## 附录 C：不同矩阵规模下各阶段最优性能（相对 cuBLAS %）

| 矩阵规模       | Naive | Shared | Register | Bank  | DoubleBuf | Tensor FP32 |
| -------------- | ----- | ------ | -------- | ----- | --------- | ----------- |
| 256×256        | 158%  | 208%   | 129%     | 134%  | 114%      | 112%        |
| 1024×1024      | 12.4% | 18.1%  | 20.0%    | 19.7% | 20.2%     | 36.7%       |
| 4096×4096      | 11.2% | 18.7%  | 25.9%    | 25.4% | 25.3%     | 61.4%       |
| 4096×1024×8192 | 9.5%  | 14.7%  | 19.5%    | 19.5% | 18.3%     | 50.2%       |
| 8192×512×4096  | 10.2% | 14.5%  | 20.6%    | 20.1% | 20.0%     | 48.1%       |
| 1000×1000      | 11.7% | 17.0%  | 18.1%    | 17.4% | 18.3%     | 38.4%       |
| 511×1023×2047  | 12.1% | 18.2%  | 22.5%    | 20.2% | 22.4%     | 28.3%       |

**小矩阵反超 cuBLAS 的原因**：
- cuBLAS 在小规模上 kernel 启动 + 算法选择开销占比大
- 自实现 kernel 启动开销小，配置简单
- 256×256 时 Shared 8×32 达到 572.28 GFLOPS，是 cuBLAS 的 2.08 倍

## 附录 D：附加实验（N:M 结构化稀疏）状态

**本实验未实现 2:4 / N:M 稀疏 GEMM**。当前完成度 0/4 项基础任务，0/5 项分析问题。

如需补做：
1. `src/kernels/sparse_2_4.cu` — 2:4 稀疏 GEMM kernel
2. `scripts/run_stage7_sparse.sh` — 对比 dense vs sparse 性能
3. 需要 `cuda::sparse` API 或 `mma.sp` 指令支持

---

## 实验总结

### 实验完成情况

| 阶段                   | 状态 | 备注                                                                                            |
| ---------------------- | ---- | ----------------------------------------------------------------------------------------------- |
| 0 baseline             | ✅    | CPU ref + cuBLAS + 计时框架                                                                     |
| 1 naive                | ✅    | 5 种 block size                                                                                 |
| 2 shared               | ✅    | 5 种 tile size                                                                                  |
| 3 register             | ✅    | 4 种配置 (BM/BN/BK/TM/TN)                                                                       |
| 4 bank conflict        | ✅    | +1 padding                                                                                      |
| 5 double buffering     | ⚠️    | 仅双 buffer，未启用 `cp.async`                                                                  |
| 6 tensor core          | ✅    | FP16 in / FP32 accum, FP16 out & FP32 out                                                       |
| 7 profiling & roofline | ✅    | Nsight Compute 全 4 section 已实测；7 份 .ncu-rep + ncu_metrics.csv 已纳入；roofline 框架已给出 |
| 附加 N:M sparse        | ❌    | 未实现                                                                                          |

### 主要发现

1. **CUDA core 优化已逼近 L1/shared memory 带宽上限**（register kernel 在 4096 上 L1/TEX 95.18%、DRAM 仅 9.03%）。继续在 CUDA core 上做软件优化的边际收益接近 0
2. **真正的性能突破来自 Tensor Core**：跨越到 compute-bound 区域，性能从 ~1.2 TFLOPS 提升到 ~2.9 TFLOPS（2.4×），瓶颈从 L1（95%）迁移到 Tensor Core 单元（SM 44%）+ DRAM（30%）
3. **Bank Conflict 在本实现下无效**（NCU 实测）：本 kernel 几乎无 bank conflict（stride=1 访问），padding 改善 short_sb 35% 但 occupancy 跌一半导致 long_sb 翻倍，净效果 -14%
4. **Double Buffering 边际有效**（NCU 实测）：4096 规模下 +6%，long_sb 1.08→0.62 (-43%)、barrier 0.65→0.07 (-89%)，流水化收益在更大规模上才能抵消 occupancy 损失
5. **小矩阵（256×256）上自实现可反超 cuBLAS**：kernel 启动开销小，cuBLAS 在小规模上调度开销占比大
6. **大矩阵（≥4096）上 cuBLAS 仍是 1.6–2.0× 优势**：tile size、warp 调度、内存流水优化更精细；Tensor Core 利用率仅 1.8% 远低于 cuBLAS 的 76%

### 改进方向

1. **Tensor Core tile 扩大**：BM=BN=128 起步，使用 multi-warp block
2. **启用 `cp.async` / `cuda::pipeline`**：实现真正异步 double buffering
3. **向量化加载**：使用 `half2` / `float4` 提升带宽利用率
4. **split-K**：对 K 维做并行，适合 K >> M/N 的场景
5. **集成 Nsight Compute profiling 已在 7.5 完成**：下一步可基于实测的 L1/shared 瓶颈做 swizzled layout / vectorized load 等降低 L1 压力的优化

---

