# 高级体系结构大实验：矩阵乘法算子的实现、

# 优化与性能分析

## 一、实验基本信息

实验主题：矩阵乘法算子的实现、优化与体系结构分析


## 二、实验主题与背景

本次大实验围绕矩阵乘法算子 GEMM（General Matrix Multiplication）展开：

其中：

A 为 M * K 矩阵，默认 row-major； B 为 K * N 矩阵，默认 row-major；C 为 M * N 矩阵，默认 row-major。

矩阵乘法是高性能计算、深度学习训练与推理中的基础算子。一个高性能 GEMM kernel 的优化过程覆 盖了 GPU 体系结构中的多个核心问题：线程组织、内存层次、数据复用、同步、流水化、专用矩阵计算 单元、精度权衡以及稀疏性利用。

体系结构问题 在 GEMM 中的体现

SIMT 并行 thread、block、warp 如何映射到输出矩阵

内存层次 global memory、shared memory、register 的数据复用

访存合并 A、B、C 的连续访问、对齐、向量化 load/store

同步与流水 __syncthreads() 、double buffering、async copy

计算单元利用 CUDA core 与 Tensor Core 的吞吐差异

精度与性能权衡 FP32、TF32、FP16、BF16、FP8、FP4

稀疏性利用 2:4 / N:M structured sparsity

本实验的重点不是调用现成高性能库，而是通过逐步实现和剖析，理解现代 GPU 上高性能 GEMM 背后 的体系结构机制。

## 三、实验目标

完成本实验后，学生应能够：

实现常规算子优化

1. 独立实现一个正确的 CUDA 矩阵乘法算子。
2. 通过 shared memory tiling、register blocking、bank conflict 优化、double buffering 等手段逐 步改进 GEMM kernel。
3. 理解 CUDA core GEMM 与 Tensor Core GEMM 的差异。

会用基本的性能分析

1. 使用 Roofline 模型解释不同优化阶段的性能变化。
2. 使用 Nsight Compute 等工具分析 kernel 的性能瓶颈。

了解其他优化方向

1. 可选探索结构化稀疏矩阵乘法，例如 2:4 或更一般的 N:M 稀疏模式。
2. 对 Tensor Core、异步访存、流水化等现代 GPU GEMM 优化思想形成整体认识。

## 四、实验环境建议

建议使用如下环境：

CUDA Toolkit； NVIDIA GPU，建议至少为 Ampere 架构； Nsight Compute； cuBLAS，用作性能和正确性参考；

## 五、实验对象与基本要求

### 5.1 基础功能要求

要求实现并优化：

基础要求如下：

1. 支持一般 、 、 尺寸；
2. 至少完整支持 alpha = 1, beta = 0 ；
3. 至少支持 FP32；
4. Tensor Core 阶段可使用 FP16 输入、FP32 累加；
5. 能正确处理非 tile 对齐尺寸；
6. 所有主要 kernel 应使用统一接口，方便测试和对比。

### 5.2 正确性要求

要求与 CPU reference 或 cuBLAS 结果比较：

FP32：相对误差控制在合理范围内 FP16 / Tensor Core：允许更宽松误差

注意：大规模矩阵可使用 cuBLAS 作为 reference；CPU reference 可主要用于小规模或中等规模测 试。

### 5.3 测试矩阵规模建议

建议至少覆盖以下三类测试。

正方矩阵

非正方矩阵

非 tile 对齐矩阵

## 六、总体实验安排

主线实验共 8 个阶段，其中阶段 0–6 侧重实现与优化，阶段 7 侧重性能分析、Roofline 建模与整体总 结。

阶段 名称 核心目的

阶段建立正确性验证、计时和 cuBLAS 实验准备与性能基线 0 baseline

阶段 朴素 CUDA GEMM 理解 thread 到输出元素的直接映射

阶段 Shared Memory 分块 GEMM 利用 block-level tiling 提高数据复用

阶段 Register Blocking / Thread-level Tiling 利用寄存器提高 thread-level 数据复用

阶段Shared Memory Bank Conflict 分析与优分析 shared memory 访问冲突并优化布 4化局

阶段 Double Buffering 与访存流水化 尝试重叠数据加载与计算

阶段 Tensor Core GEMM 使用专用矩阵计算单元加速 GEMM

阶段 Profiling、Roofline 建模与整体总结 使用性能工具和模型解释所有优化阶段

附加实验：N:M 结构化稀疏矩阵乘法。

## 七、各阶段实验内容

### 阶段 0：实验准备与性能基线

目标

建立后续所有实验共用的正确性验证、计时和性能统计框架。

要求

1. 实现 CPU reference GEMM。

2. 使用 CUDA Event 测量 kernel 时间。

3. 调用 cuBLAS cublasSgemm 作为参考。

4. 编写统一 benchmark 驱动程序，支持输入 、 、 。

5. 输出以下指标：

kernel 平均运行时间； GFLOPS / TFLOPS； 相对 cuBLAS 的性能比例； 正确性误差。

性能计算公式：

注：公式默认 t 单位是秒，而后续性能分析使用CUDA Event 通常得到的可能是毫秒。

### 阶段 1：朴素 CUDA GEMM

目标

实现最基础的 CUDA GEMM kernel，理解线程到输出矩阵元素的直接映射。

基本思想

每个 thread 计算 中的一个元素：

int row = blockIdx.y * blockDim.y + threadIdx.y; int col = blockIdx.x * blockDim.x + threadIdx.x;

float acc = 0.0f; for (int k = 0; k < K; ++k) { acc += A[row * K + k] * B[k * N + col]; } C[row * N + col] = acc;

要求

1. 实现 naive CUDA GEMM。
2. 支持一般 、 、 。
3. 尝试不同 block 配置，例如 、 、 。
4. 比较不同 block size 下的性能差异。
5. 分析 global memory 访问模式。

需要回答的问题

1. 为什么 naive kernel 性能较差？
2. A 和 B 的访存模式分别是什么？
3. 哪些访问是连续的，哪些不是？
4. 每个 A/B 元素被复用了多少次？
5. 该 kernel 更接近 memory-bound 还是 compute-bound？

### 阶段 2：Shared Memory 分块 GEMM

目标

通过 block-level tiling 将 A 和 B 的局部数据搬入 shared memory，提高数据复用率。

基本思想

将输出矩阵 C 划分为多个 tile，每个 thread block 负责一个 C tile。沿 K 维度循环时：

1. 从 global memory 加载一个 A tile；
2. 从 global memory 加载一个 B tile；
3. 将两个 tile 存入 shared memory；
4. block 内线程复用 shared memory 中的数据完成部分累加；
5. 进入下一轮 K tile。

要求

1. 实现 shared memory tiled GEMM。
2. 正确处理边界尺寸。
3. 正确使用 __syncthreads() 。
4. 尝试不同 tile size。
5. 比较 naive 与 shared memory tiled 版本的性能差异。

需要回答的问题

1. shared memory tiling 为什么能带来性能提升？
2. A tile 与 B tile 的数据复用分别体现在哪里？
3. tile size 增大为什么不一定总是更快？
4. shared memory 占用如何影响 occupancy？
5. 优化后 global memory 重复读取是否减少？如何证明？

### 阶段 3：Register Blocking / Thread-level Tiling

目标

让一个 thread 计算多个输出元素，利用寄存器进一步提高数据复用，减少 shared memory 访问次数。

基本思想

在 shared memory tiled kernel 中，一个 thread 通常计算一个输出元素。本阶段要求一个 thread 计 算一个小的输出子块，例如：

每个 thread 维护多个 accumulator，从 shared memory 中读取 A/B 小片段，在寄存器中完成多次乘 加。

推荐分层概念

Block tile：BM * BN
K tile： BK
Thread tile：TM * TN

可尝试配置：

配置 BM BN BK TM TN

A 64 64 8 4 4

B 64 128 8 4 8

C 128 128 8 8 8

D 128 128 16 8 8

要求

1. 实现 register-blocked GEMM。
2. 让一个 thread 维护多个 accumulator。
3. 尝试不同 thread tile 大小。
4. 观察 register usage 与性能变化。
5. 分析 occupancy 与寄存器压力之间的关系。

需要回答的问题

1. 为什么 thread-level tiling 能进一步提升性能？
2. 一个 thread 计算多个输出元素时，A/B 数据如何在寄存器中复用？
3. 为什么 shared memory 访问次数可能减少？
4. 为什么寄存器过多会降低 occupancy？
5. 当前 kernel 的主要瓶颈是否发生变化？

### 阶段 4：Shared Memory Bank Conflict 分析与优化

目标

分析 register-blocked kernel 中 shared memory 的访问模式，理解 bank conflict 的成因，并通过 padding 或 layout 变换进行优化。

背景

shared memory 被划分为多个 bank。如果一个 warp 内多个线程访问映射到同一个 bank 的不同地 址，则可能发生 bank conflict，导致访问串行化。

常见优化方式包括：

1. 在 shared memory tile 尾部添加 padding；
2. 改变 A/B tile 在 shared memory 中的存储布局；
3. 对 B tile 采用转置存储；
4. 调整 thread mapping。

要求

1. 分析当前 register-blocked kernel 的 shared memory 访问模式。
2. 判断 A tile 或 B tile 的哪些访问可能产生 bank conflict。
3. 使用 Nsight Compute 观察 shared memory 相关指标。
4. 至少实现一种优化方法，例如 padding 或 layout 变换。
5. 比较优化前后的性能和 profiling 指标。

需要回答的问题

1. 一个 warp 内线程如何访问 shared memory？
2. 哪些访问模式容易发生 bank conflict？
3. padding 为什么可能有效？
4. padding 会带来什么代价？
5. shared memory 使用量增加后是否影响 occupancy？
6. bank conflict 降低后性能是否一定提升？为什么？
7. 当前 kernel 中，A tile 和 B tile 哪一个更可能是主要冲突来源？

### 阶段 5：Double Buffering 与访存流水化

目标

通过 double buffering 将“下一块 tile 的加载”与“当前 tile 的计算”重叠，尝试隐藏部分访存延迟。

基本思想

普通 tiled GEMM 的主循环通常为：

1. load 当前 tile；
2. 同步；
3. compute 当前 tile；
4. 同步；
5. 进入下一轮。

double buffering 使用两组 shared memory buffer：

一组 buffer 用于当前计算； 另一组 buffer 用于预取下一轮 tile； 两组 buffer 交替使用，形成流水。

可进一步尝试：

cp.async ； cuda::pipeline ； 异步 copy 与同步控制。

要求

1. 实现 single-buffer baseline。
2. 实现 double-buffer 版本。
3. 比较 single-buffer 与 double-buffer 的性能差异。
4. 分析 shared memory 占用增加带来的影响。
5. 可选：尝试 cp.async 或 cuda::pipeline 。

需要回答的问题

1. double buffering 试图隐藏什么延迟？
2. 为什么需要两组 shared memory buffer？
3. double buffering 增加了多少 shared memory 使用量？
4. 在什么情况下 double buffering 收益更明显？
5. double buffering 与 padding 是否会相互影响？
6. 如果 double buffering 没有带来性能提升，可能原因是什么？

### 阶段 6：Tensor Core GEMM

目标

引入专用矩阵乘法硬件单元，理解 Tensor Core GEMM 的编程方法、性能收益和精度代价。

推荐实现

实现一个基于 nvcuda::wmma 的 GEMM：

A: FP16 B: FP16 Accumulator: FP32 C: FP16 或 FP32

需要使用：

1. wmma::fragment ；
2. wmma::load_matrix_sync ；
3. wmma::mma_sync ；
4. wmma::store_matrix_sync 。

也可以使用 CUTLASS 或 cuBLASLt 完成 Tensor Core 参数实验，并在报告中说明。

要求

1. 完成一个 Tensor Core 相关实验。
2. 对比 CUDA core kernel 与 Tensor Core kernel。
3. 明确说明数据类型和精度设置。
4. 分析性能提升来源和精度差异来源。

需要回答的问题

1. Tensor Core 为什么比普通 CUDA core 更适合矩阵乘法？
2. WMMA 中 fragment 的作用是什么？
3. Tensor Core 版本是否仍然需要 shared memory staging？
4. 为什么 Tensor Core 常与低精度类型结合？
5. 性能提升的代价是什么？

### 阶段 7：Profiling、Roofline 建模与整体总结

目标

本阶段不要求继续增加新的 kernel，而是要求对前面各个实现版本进行系统性分析。学生需要使用 profiling 工具(Nsight Compute)和性能模型解释各阶段的优化收益、瓶颈迁移和资源代价。

Profiing要求

本阶段要求对所有主要 kernel 版本进行总体性能分析，包括 naive、shared memory tiled、register blocked、bank conflict optimized、double-buffered、Tensor Core。

其中，Nsight Compute 的详细 profiling 至少需要覆盖以下两个阶段：

1. 阶段 4：Shared Memory Bank Conflict 分析与优化；
2. 阶段 5：Double Buffering 与访存流水化。

Profiling 指标建议

其他可能用到的通用性能指标如下。不要求每个阶段都给出所有指标，只需要在必要的优化阶段给出有 代表性的指标即可。

指标 作用

Kernel execution time 直接运行时间

Achieved FLOPS / TFLOPS 实际计算吞吐

SM throughput SM 计算资源利用率

Memory throughput global memory 带宽利用

L2 hit rate L2 cache 命中情况

Achieved occupancy 实际 occupancy

Register usage 每线程寄存器使用量

Shared memory usage 每 block shared memory 使用量

Warp stall reasons 判断 stall 来源

“阶段4：Shared Memory Bank Conflict” 必须考虑的相关指标：

指标 作用

Shared load/store transactions shared memory 访问事务数量

Shared memory bank conflicts 判断 bank conflict

Shared memory throughput shared memory 吞吐

Shared memory replay / conflict 指标 判断访问是否被重复执行

”阶段5：Double Buffering 与访存流水化“的相关分析：

single-buffer 与 double-buffer 的 kernel time 对比； long scoreboard stall 是否下降； barrier stall 是否变化； shared memory 使用量是否导致 occupancy 下降； 使用 cp.async / cuda::pipeline 时，是否观察到 load-compute overlap 的收益。

使用 profiling 工具和性能模型进行体系结构解释

1. 每一代 kernel 的性能瓶颈是什么？

2. 优化后瓶颈是否发生迁移？

3. 当前实现距离硬件理论上限还有多远？

4. 性能提升来自访存优化、计算复用、并行度提升，还是 Tensor Core 使用？

5. 哪些优化在自己的实现中有效，哪些没有达到预期？

6. 如何用 Roofline 模型解释这些结果？

Roofline 模型用于判断一个 kernel 当前主要受限于：计算能力；内存带宽；或者两者之外的 其他因素，例如同步、bank conflict、occupancy、指令调度。 Roofline 的横轴是算术强度： 纵轴是实际性能：

## 八、附加实验

附加实验不是基础要求，只提供少量加分。

背景论文：Learning N:M Fine-grained Structured Sparse Neural Networks From Scratch。

基础任务：2:4 稀疏 GEMM

要求：

1. 对矩阵 A 或 B 做 2:4 剪枝，使得矩阵中一些位置置0，且存在规律：

每连续 4 个元素中保留 2 个； 另外 2 个置零； 可保留最大绝对值的 2 个元素。
2. 设计压缩格式：

nonzero values； metadata； 原始 shape； layout 描述。
3. 实现一个简单 sparse GEMM kernel。

4. 与 dense GEMM 比较：

理论 FLOPs 减少； 实际性能变化； metadata 开销； 精度损失。

进阶任务：一般 N:M 稀疏

可推广到：

N:M = 1:2, 2:4, 4:8

需要分析：

问题 要求

压缩率 values + metadata 后真实压缩率是多少

访存模式 metadata 是否破坏 coalescing

负载均衡 每个 tile 的有效非零数是否均匀

Tensor Core 适配 哪些模式能映射到硬件 sparse MMA

精度影响 pruning 策略对误差的影响

需要回答的问题

1. 为什么“理论计算量减半”不一定等于“实际速度翻倍”？
2. metadata 会带来哪些额外代价？
3. 稀疏存储是否破坏访存连续性？
4. 稀疏矩阵乘法的负载是否均衡？
5. 哪些 N:M 模式更可能匹配硬件加速能力？

## 九、实验报告要求

内容包括以下部分。

### 9.1 实现概述

简要说明每个 kernel 的设计：

Kernel 0: CPU reference Kernel 1: naive CUDA Kernel 2: shared memory tiled Kernel 3: register blocked Kernel 4: bank conflict optimized Kernel 5: double-buffered Kernel 6: Tensor Core Kernel 7: sparse bonus, if any

需要回答前面各阶段中的“需要回答的问题”。

### 9.2 正确性验证

至少包含：

测试矩阵规模； 误差统计； 边界尺寸处理；

### 9.3 性能结果

每个阶段应给出：

运行时间； GFLOPS / TFLOPS； 相对上一版本提升(不一定每阶段都会有提升，如实记录分析即可) 相对 cuBLAS 的比例。

### 9.4 Profiling 与体系结构解释

完成阶段7中的性能分析以及回答最后的问题。

## 十、评分标准

1. 最终性能数值不作为直接评分依据。 不以固定 TFLOPS、相对 cuBLAS 百分比或排名作为给分标 准。
2. 正确性影响评分。 若某阶段 kernel 结果不正确，则该阶段不能获得完整分数。
3. 评分重点是分析能力。 主要考察是否能够解释每一阶段为什么这样优化、解决了什么问题、带来了 什么收益、引入了什么代价，以及 profiling 指标是否支持结论。
4. 优化失败也可以得分。 如果某个优化没有带来性能提升，但能用数据和体系结构机制合理解释，仍 可获得较高分数。
5. 附加实验有少量加分。 附加题不是基础得分的必要条件。

## 十一、最终提交要求

每组提交一份实验报告（PDF）和一个代码压缩包。代码压缩包只需包含最终完成所有优化后的矩阵乘 法代码版本；不强制提交每个阶段的独立代码文件。实验报告中仍需包含各阶段的实现思路、性能数 据、profiling 分析和问题回答。附加实验如完成，需一并提交相应代码或说明材料。

## 十二、参考资料

1. NVIDIA CUDA C++ Programming Guide https://docs.nvidia.com/cuda/cuda-c-programming-guide/
2. NVIDIA CUDA Programming Guide, Warp Matrix Functions / WMMA https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions
3. NVIDIA CUTLASS GitHub Repository https://github.com/NVIDIA/cutlass
4. NVIDIA cuSPARSELt Documentation https://docs.nvidia.com/cuda/cusparselt/
5. NVIDIA cuBLAS Documentation https://docs.nvidia.com/cuda/cublas/