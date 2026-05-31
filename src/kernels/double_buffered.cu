#include "gemm_common.h"

// ========== Stage 5: Double Buffering GEMM ==========
template <int BM, int BN, int BK, int TM, int TN>
__global__ void double_buffer_gemm_kernel(float *C, const float *A, const float *B,
                                          int M, int N, int K,
                                          float alpha, float beta)
{
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    int C_row = block_row + thread_row * TM;
    int C_col = block_col + thread_col * TN;

    float acc[TM][TN] = {0.0f};

    int numTiles = (K + BK - 1) / BK;

    // ========== 预加载 tile 0 ==========
    int write_stage = 0;

    // 加载 A tile 0
    int A_col = 0;
#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        int row = C_row + i;
#pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            int col = A_col + k;
            if (row < M && col < K)
                As[write_stage][thread_row * TM + i][k] = A[row * K + col];
            else
                As[write_stage][thread_row * TM + i][k] = 0.0f;
        }
    }

    // 加载 B tile 0
    int B_row = 0;
#pragma unroll
    for (int j = 0; j < TN; ++j)
    {
        int col = C_col + j;
#pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            int row = B_row + k;
            if (row < K && col < N)
                Bs[write_stage][k][thread_col * TN + j] = B[row * N + col];
            else
                Bs[write_stage][k][thread_col * TN + j] = 0.0f;
        }
    }

    __syncthreads();
    int read_stage = write_stage;
    write_stage = 1 - write_stage;

    // ========== 主循环 ==========
    for (int tile = 0; tile < numTiles; ++tile)
    {
        // 异步加载下一个 tile（如果存在）
        if (tile + 1 < numTiles)
        {
            int next_A_col = (tile + 1) * BK;
            int next_B_row = (tile + 1) * BK;

#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
                int row = C_row + i;
#pragma unroll
                for (int k = 0; k < BK; ++k)
                {
                    int col = next_A_col + k;
                    if (row < M && col < K)
                        As[write_stage][thread_row * TM + i][k] = A[row * K + col];
                    else
                        As[write_stage][thread_row * TM + i][k] = 0.0f;
                }
            }

#pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                int col = C_col + j;
#pragma unroll
                for (int k = 0; k < BK; ++k)
                {
                    int row = next_B_row + k;
                    if (row < K && col < N)
                        Bs[write_stage][k][thread_col * TN + j] = B[row * N + col];
                    else
                        Bs[write_stage][k][thread_col * TN + j] = 0.0f;
                }
            }
        }

// 计算当前 tile（使用 read_stage）
#pragma unroll
        for (int k = 0; k < BK; ++k)
        {
#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
                float a_val = As[read_stage][thread_row * TM + i][k];
#pragma unroll
                for (int j = 0; j < TN; ++j)
                {
                    acc[i][j] += a_val * Bs[read_stage][k][thread_col * TN + j];
                }
            }
        }

        // 同步并切换 buffer（除了最后一次）
        if (tile + 1 < numTiles)
        {
            __syncthreads();
            read_stage = write_stage;
            write_stage = 1 - write_stage;
        }
    }

// ========== 写回结果 ==========
#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        int row = C_row + i;
        if (row >= M)
            continue;
#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            int col = C_col + j;
            if (col < N)
            {
                C[row * N + col] = alpha * acc[i][j] + beta * C[row * N + col];
            }
        }
    }
}

// ========== 包装宏 ==========
#define LAUNCH_DOUBLE_BUFFER_KERNEL(BM, BN, BK, TM, TN)                        \
    do                                                                         \
    {                                                                          \
        dim3 block(BN / TN, BM / TM);                                          \
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);                       \
        double_buffer_gemm_kernel<BM, BN, BK, TM, TN>                          \
            <<<grid, block, 0, stream>>>(d_C, d_A, d_B, M, N, K, alpha, beta); \
    } while (0)

// ========== 启动函数 ==========
void run_double_buffer_kernel(float *d_C, const float *d_A, const float *d_B,
                              int M, int N, int K,
                              float alpha, float beta,
                              cudaStream_t stream, int config)
{
    switch (config)
    {
    case 1:
        LAUNCH_DOUBLE_BUFFER_KERNEL(64, 64, 8, 4, 4); // Config A: 小块配置
        break;
    case 2:
        LAUNCH_DOUBLE_BUFFER_KERNEL(64, 128, 8, 4, 8); // Config B: 大列配置
        break;
    case 3:
        LAUNCH_DOUBLE_BUFFER_KERNEL(128, 128, 8, 8, 8); // Config C: 大块配置 BK=8
        break;
    case 4:
        LAUNCH_DOUBLE_BUFFER_KERNEL(128, 128, 16, 8, 8); // Config D: 大块配置 BK=16
        break;
    default:
        LAUNCH_DOUBLE_BUFFER_KERNEL(128, 128, 8, 8, 8);
        break;
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}
