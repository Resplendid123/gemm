#include "gemm_common.h"

// ========== Stage 2: Shared Memory Tiled GEMM ==========
template <int BM, int BN, int BK>
__global__ void shared_memory_gemm_kernel(float *C, const float *A, const float *B,
                                          int M, int N, int K,
                                          float alpha, float beta)
{
    // 共享内存声明
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int row = threadIdx.y;
    int col = threadIdx.x;

    float acc = 0.0f;

    int numTiles = (K + BK - 1) / BK;

    for (int tile = 0; tile < numTiles; ++tile)
    {
        // ========== 加载 A tile 到共享内存 ==========
        // 线程 (row, col) 加载 As[row][col]
        int A_row = block_row + row;
        int A_col = tile * BK + col;
        if (A_row < M && A_col < K)
            As[row][col] = A[A_row * K + A_col];
        else
            As[row][col] = 0.0f;

        // ========== 加载 B tile 到共享内存 ==========
        // 线程 (row, col) 加载 Bs[row][col]
        int B_row = tile * BK + row;
        int B_col = block_col + col;
        if (B_row < K && B_col < N)
            Bs[row][col] = B[B_row * N + B_col];
        else
            Bs[row][col] = 0.0f;

        __syncthreads();

        // ========== 计算当前 tile ==========
#pragma unroll
        for (int k = 0; k < BK; ++k)
            acc += As[row][k] * Bs[k][col];

        __syncthreads();
    }

    // ========== 写回结果 ==========
    int C_row = block_row + row;
    int C_col = block_col + col;
    if (C_row < M && C_col < N)
        C[C_row * N + C_col] = alpha * acc + beta * C[C_row * N + C_col];
}

// ========== 包装宏，统一启动方式 ==========
#define LAUNCH_SHARED_KERNEL(BM, BN, BK)                                       \
    do                                                                         \
    {                                                                          \
        dim3 block(BN, BM);                                                    \
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);                       \
        shared_memory_gemm_kernel<BM, BN, BK>                                  \
            <<<grid, block, 0, stream>>>(d_C, d_A, d_B, M, N, K, alpha, beta); \
    } while (0)

// ========== 启动函数，使用 switch-case 选择配置 ==========
void run_shared_memory_kernel(float *d_C, const float *d_A, const float *d_B,
                              int M, int N, int K,
                              float alpha, float beta,
                              cudaStream_t stream,
                              int config)
{
    switch (config)
    {
    case 1:
        LAUNCH_SHARED_KERNEL(8, 8, 8); // Config A: 8x8 小块
        break;
    case 2:
        LAUNCH_SHARED_KERNEL(16, 16, 16); // Config B: 16x16 中块
        break;
    case 3:
        LAUNCH_SHARED_KERNEL(32, 32, 32); // Config C: 32x32 大块
        break;
    case 4:
        LAUNCH_SHARED_KERNEL(32, 8, 32); // Config D: 32x8 非方形
        break;
    case 5:
        LAUNCH_SHARED_KERNEL(8, 32, 32); // Config E: 8x32 非方形
        break;
    default:
        LAUNCH_SHARED_KERNEL(16, 16, 16); // 默认: 16x16
        break;
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}
