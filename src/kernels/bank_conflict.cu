#include "gemm_common.h"

// ========== Stage 4: Register Blocking + Bank Conflict Avoidance ==========
// 基于 Stage 3 的 Register Blocking GEMM，添加 shared memory padding 避免 bank conflict

// ========== Config 1: BM=64, BN=64, BK=8, TM=4, TN=4 ==========
#define BM1 64
#define BN1 64
#define BK1 8
#define TM1 4
#define TN1 4

// Shared memory with padding to avoid bank conflicts
__shared__ float As1[BM1][BK1 + 1];
__shared__ float Bs1[BK1][BN1 + 1];

__global__ void bank_gemm_kernel_1(float *C, const float *A, const float *B,
                                   int M, int N, int K,
                                   float alpha, float beta)
{
    int block_row = blockIdx.y * BM1;
    int block_col = blockIdx.x * BN1;

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    int C_row = block_row + thread_row * TM1;
    int C_col = block_col + thread_col * TN1;

    float acc[TM1][TN1];
#pragma unroll
    for (int i = 0; i < TM1; ++i)
        for (int j = 0; j < TN1; ++j)
            acc[i][j] = 0.0f;

    int numTiles = (K + BK1 - 1) / BK1;

    for (int tile = 0; tile < numTiles; ++tile)
    {
        // Load A tile
        int A_col = tile * BK1;
#pragma unroll
        for (int i = 0; i < TM1; ++i)
        {
            int A_row = C_row + i;
#pragma unroll
            for (int k = 0; k < BK1; ++k)
            {
                if (A_row < M && (A_col + k) < K)
                    As1[thread_row * TM1 + i][k] = A[A_row * K + (A_col + k)];
                else
                    As1[thread_row * TM1 + i][k] = 0.0f;
            }
        }

        // Load B tile
        int B_row = tile * BK1;
#pragma unroll
        for (int j = 0; j < TN1; ++j)
        {
            int B_col = C_col + j;
#pragma unroll
            for (int k = 0; k < BK1; ++k)
            {
                if ((B_row + k) < K && B_col < N)
                    Bs1[k][thread_col * TN1 + j] = B[(B_row + k) * N + B_col];
                else
                    Bs1[k][thread_col * TN1 + j] = 0.0f;
            }
        }

        __syncthreads();

        // Compute
#pragma unroll
        for (int k = 0; k < BK1; ++k)
        {
#pragma unroll
            for (int i = 0; i < TM1; ++i)
            {
#pragma unroll
                for (int j = 0; j < TN1; ++j)
                {
                    acc[i][j] += As1[thread_row * TM1 + i][k] *
                                 Bs1[k][thread_col * TN1 + j];
                }
            }
        }

        __syncthreads();
    }

    // Write back
#pragma unroll
    for (int i = 0; i < TM1; ++i)
    {
#pragma unroll
        for (int j = 0; j < TN1; ++j)
        {
            int row = C_row + i;
            int col = C_col + j;
            if (row < M && col < N)
                C[row * N + col] = alpha * acc[i][j] + beta * C[row * N + col];
        }
    }
}

// ========== Config 2: BM=64, BN=128, BK=8, TM=4, TN=8 ==========
#define BM2 64
#define BN2 128
#define BK2 8
#define TM2 4
#define TN2 8

__shared__ float As2[BM2][BK2 + 1];
__shared__ float Bs2[BK2][BN2 + 1];

__global__ void bank_gemm_kernel_2(float *C, const float *A, const float *B,
                                   int M, int N, int K,
                                   float alpha, float beta)
{
    int block_row = blockIdx.y * BM2;
    int block_col = blockIdx.x * BN2;

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    int C_row = block_row + thread_row * TM2;
    int C_col = block_col + thread_col * TN2;

    float acc[TM2][TN2];
#pragma unroll
    for (int i = 0; i < TM2; ++i)
        for (int j = 0; j < TN2; ++j)
            acc[i][j] = 0.0f;

    int numTiles = (K + BK2 - 1) / BK2;

    for (int tile = 0; tile < numTiles; ++tile)
    {
        int A_col = tile * BK2;
#pragma unroll
        for (int i = 0; i < TM2; ++i)
        {
            int A_row = C_row + i;
#pragma unroll
            for (int k = 0; k < BK2; ++k)
            {
                if (A_row < M && (A_col + k) < K)
                    As2[thread_row * TM2 + i][k] = A[A_row * K + (A_col + k)];
                else
                    As2[thread_row * TM2 + i][k] = 0.0f;
            }
        }

        int B_row = tile * BK2;
#pragma unroll
        for (int j = 0; j < TN2; ++j)
        {
            int B_col = C_col + j;
#pragma unroll
            for (int k = 0; k < BK2; ++k)
            {
                if ((B_row + k) < K && B_col < N)
                    Bs2[k][thread_col * TN2 + j] = B[(B_row + k) * N + B_col];
                else
                    Bs2[k][thread_col * TN2 + j] = 0.0f;
            }
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK2; ++k)
        {
#pragma unroll
            for (int i = 0; i < TM2; ++i)
            {
#pragma unroll
                for (int j = 0; j < TN2; ++j)
                {
                    acc[i][j] += As2[thread_row * TM2 + i][k] *
                                 Bs2[k][thread_col * TN2 + j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM2; ++i)
    {
#pragma unroll
        for (int j = 0; j < TN2; ++j)
        {
            int row = C_row + i;
            int col = C_col + j;
            if (row < M && col < N)
                C[row * N + col] = alpha * acc[i][j] + beta * C[row * N + col];
        }
    }
}

// ========== Config 3: BM=128, BN=128, BK=8, TM=8, TN=8 ==========
#define BM3 128
#define BN3 128
#define BK3 8
#define TM3 8
#define TN3 8

__shared__ float As3[BM3][BK3 + 1];
__shared__ float Bs3[BK3][BN3 + 1];

__global__ void bank_gemm_kernel_3(float *C, const float *A, const float *B,
                                   int M, int N, int K,
                                   float alpha, float beta)
{
    int block_row = blockIdx.y * BM3;
    int block_col = blockIdx.x * BN3;

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    int C_row = block_row + thread_row * TM3;
    int C_col = block_col + thread_col * TN3;

    float acc[TM3][TN3];
#pragma unroll
    for (int i = 0; i < TM3; ++i)
        for (int j = 0; j < TN3; ++j)
            acc[i][j] = 0.0f;

    int numTiles = (K + BK3 - 1) / BK3;

    for (int tile = 0; tile < numTiles; ++tile)
    {
        int A_col = tile * BK3;
#pragma unroll
        for (int i = 0; i < TM3; ++i)
        {
            int A_row = C_row + i;
#pragma unroll
            for (int k = 0; k < BK3; ++k)
            {
                if (A_row < M && (A_col + k) < K)
                    As3[thread_row * TM3 + i][k] = A[A_row * K + (A_col + k)];
                else
                    As3[thread_row * TM3 + i][k] = 0.0f;
            }
        }

        int B_row = tile * BK3;
#pragma unroll
        for (int j = 0; j < TN3; ++j)
        {
            int B_col = C_col + j;
#pragma unroll
            for (int k = 0; k < BK3; ++k)
            {
                if ((B_row + k) < K && B_col < N)
                    Bs3[k][thread_col * TN3 + j] = B[(B_row + k) * N + B_col];
                else
                    Bs3[k][thread_col * TN3 + j] = 0.0f;
            }
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK3; ++k)
        {
#pragma unroll
            for (int i = 0; i < TM3; ++i)
            {
#pragma unroll
                for (int j = 0; j < TN3; ++j)
                {
                    acc[i][j] += As3[thread_row * TM3 + i][k] *
                                 Bs3[k][thread_col * TN3 + j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM3; ++i)
    {
#pragma unroll
        for (int j = 0; j < TN3; ++j)
        {
            int row = C_row + i;
            int col = C_col + j;
            if (row < M && col < N)
                C[row * N + col] = alpha * acc[i][j] + beta * C[row * N + col];
        }
    }
}

// ========== Config 4: BM=128, BN=128, BK=16, TM=8, TN=8 ==========
#define BM4 128
#define BN4 128
#define BK4 16
#define TM4 8
#define TN4 8

__shared__ float As4[BM4][BK4 + 1];
__shared__ float Bs4[BK4][BN4 + 1];

__global__ void bank_gemm_kernel_4(float *C, const float *A, const float *B,
                                   int M, int N, int K,
                                   float alpha, float beta)
{
    int block_row = blockIdx.y * BM4;
    int block_col = blockIdx.x * BN4;

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    int C_row = block_row + thread_row * TM4;
    int C_col = block_col + thread_col * TN4;

    float acc[TM4][TN4];
#pragma unroll
    for (int i = 0; i < TM4; ++i)
        for (int j = 0; j < TN4; ++j)
            acc[i][j] = 0.0f;

    int numTiles = (K + BK4 - 1) / BK4;

    for (int tile = 0; tile < numTiles; ++tile)
    {
        int A_col = tile * BK4;
#pragma unroll
        for (int i = 0; i < TM4; ++i)
        {
            int A_row = C_row + i;
#pragma unroll
            for (int k = 0; k < BK4; ++k)
            {
                if (A_row < M && (A_col + k) < K)
                    As4[thread_row * TM4 + i][k] = A[A_row * K + (A_col + k)];
                else
                    As4[thread_row * TM4 + i][k] = 0.0f;
            }
        }

        int B_row = tile * BK4;
#pragma unroll
        for (int j = 0; j < TN4; ++j)
        {
            int B_col = C_col + j;
#pragma unroll
            for (int k = 0; k < BK4; ++k)
            {
                if ((B_row + k) < K && B_col < N)
                    Bs4[k][thread_col * TN4 + j] = B[(B_row + k) * N + B_col];
                else
                    Bs4[k][thread_col * TN4 + j] = 0.0f;
            }
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK4; ++k)
        {
#pragma unroll
            for (int i = 0; i < TM4; ++i)
            {
#pragma unroll
                for (int j = 0; j < TN4; ++j)
                {
                    acc[i][j] += As4[thread_row * TM4 + i][k] *
                                 Bs4[k][thread_col * TN4 + j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM4; ++i)
    {
#pragma unroll
        for (int j = 0; j < TN4; ++j)
        {
            int row = C_row + i;
            int col = C_col + j;
            if (row < M && col < N)
                C[row * N + col] = alpha * acc[i][j] + beta * C[row * N + col];
        }
    }
}

// ========== 启动函数 ==========
#define LAUNCH_BANK_KERNEL(config_num)                                         \
    do                                                                        \
    {                                                                         \
        dim3 block(BN##config_num / TN##config_num, BM##config_num / TM##config_num); \
        dim3 grid((N + BN##config_num - 1) / BN##config_num, (M + BM##config_num - 1) / BM##config_num); \
        bank_gemm_kernel_##config_num<<<grid, block, 0, stream>>>(d_C, d_A, d_B, M, N, K, alpha, beta); \
    } while (0)

void run_bank_conflict_kernel(float *d_C, const float *d_A, const float *d_B,
                              int M, int N, int K,
                              float alpha, float beta,
                              cudaStream_t stream, int config)
{
    switch (config)
    {
    case 1:
        LAUNCH_BANK_KERNEL(1);
        break;
    case 2:
        LAUNCH_BANK_KERNEL(2);
        break;
    case 3:
        LAUNCH_BANK_KERNEL(3);
        break;
    case 4:
        LAUNCH_BANK_KERNEL(4);
        break;
    default:
        LAUNCH_BANK_KERNEL(1);
        break;
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}
