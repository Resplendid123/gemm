#include "gemm_common.h"

// ========== Stage 3: Register Blocking GEMM ==========
template <int BM, int BN, int BK, int TM, int TN>
__global__ void register_blocking_gemm_kernel(float *C, const float *A, const float *B,
                                              int M, int N, int K,
                                              float alpha, float beta)
{
    // 每个 block 处理的 tile 大小
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 每个 block 有 (BM/TM) * (BN/TN) 个线程
    // threadIdx.x 对应输出矩阵的列方向，threadIdx.y 对应行方向
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // 当前线程负责的 C 子块起始位置
    int thread_row_in_block = threadIdx.y;
    int thread_col_in_block = threadIdx.x;

    int C_row = block_row + thread_row_in_block * TM;
    int C_col = block_col + thread_col_in_block * TN;

    // 寄存器累加器：每个线程维护 TM×TN 个累加器
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    // 沿 K 维度分块循环
    int numTiles = (K + BK - 1) / BK;

    for (int tile = 0; tile < numTiles; ++tile)
    {
        // ========== 加载 A tile 到共享内存 ==========
        int A_col = tile * BK;
#pragma unroll
        for (int i = 0; i < TM; ++i)
        {
            int A_row = C_row + i;
#pragma unroll
            for (int k = 0; k < BK; ++k)
            {
                if (A_row < M && (A_col + k) < K)
                    As[thread_row_in_block * TM + i][k] = A[A_row * K + (A_col + k)];
                else
                    As[thread_row_in_block * TM + i][k] = 0.0f;
            }
        }

        // ========== 加载 B tile 到共享内存 ==========
        int B_row = tile * BK;
#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            int B_col = C_col + j;
#pragma unroll
            for (int k = 0; k < BK; ++k)
            {
                if ((B_row + k) < K && B_col < N)
                    Bs[k][thread_col_in_block * TN + j] = B[(B_row + k) * N + B_col];
                else
                    Bs[k][thread_col_in_block * TN + j] = 0.0f;
            }
        }

        __syncthreads();

        // ========== 计算当前 tile ==========
#pragma unroll
        for (int k = 0; k < BK; ++k)
        {
#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
#pragma unroll
                for (int j = 0; j < TN; ++j)
                {
                    acc[i][j] += As[thread_row_in_block * TM + i][k] *
                                 Bs[k][thread_col_in_block * TN + j];
                }
            }
        }

        __syncthreads();
    }

    // ========== 写回结果 ==========
#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            int row = C_row + i;
            int col = C_col + j;
            if (row < M && col < N)
                C[row * N + col] = alpha * acc[i][j] + beta * C[row * N + col];
        }
    }
}

// ========== 包装函数，使用宏来实例化模板 ==========
#define LAUNCH_REGISTER_KERNEL(BM, BN, BK, TM, TN)                             \
    do                                                                         \
    {                                                                          \
        dim3 block(BN / TN, BM / TM);                                          \
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);                       \
        register_blocking_gemm_kernel<BM, BN, BK, TM, TN>                      \
            <<<grid, block, 0, stream>>>(d_C, d_A, d_B, M, N, K, alpha, beta); \
    } while (0)

void run_register_blocking_kernel(float *d_C, const float *d_A, const float *d_B,
                                  int M, int N, int K,
                                  float alpha, float beta,
                                  cudaStream_t stream,
                                  int config)
{
    switch (config)
    {
    case 1:
        LAUNCH_REGISTER_KERNEL(64, 64, 8, 4, 4); // Config A: 小块配置，适合小矩阵
        break;
    case 2:
        LAUNCH_REGISTER_KERNEL(64, 128, 8, 4, 8); // Config B: 大列配置，适合列优先
        break;
    case 3:
        LAUNCH_REGISTER_KERNEL(128, 128, 8, 8, 8); // Config C: 大块配置，BK=8
        break;
    case 4:
        LAUNCH_REGISTER_KERNEL(128, 128, 16, 8, 8); // Config D: 大块配置，BK=16
        break;
    default:
        LAUNCH_REGISTER_KERNEL(64, 64, 8, 4, 4);
        break;
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}
