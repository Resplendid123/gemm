#include "gemm_common.h"

// ========== Stage 2: Shared Memory Tiled GEMM ==========
// 支持多种 tile size: 8, 16, 32

// ========== TILE_SIZE = 8 ==========
#define TILE_SIZE_8 8
__shared__ float As_8[TILE_SIZE_8][TILE_SIZE_8];
__shared__ float Bs_8[TILE_SIZE_8][TILE_SIZE_8];

__global__ void shared_memory_gemm_kernel_8(float *C, const float *A, const float *B,
                                            int M, int N, int K,
                                            float alpha, float beta)
{
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    float acc = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE_8 - 1) / TILE_SIZE_8; ++tile)
    {
        int A_row = block_row * TILE_SIZE_8 + row;
        int A_col = tile * TILE_SIZE_8 + col;
        if (A_row < M && A_col < K)
            As_8[row][col] = A[A_row * K + A_col];
        else
            As_8[row][col] = 0.0f;

        int B_row = tile * TILE_SIZE_8 + row;
        int B_col = block_col * TILE_SIZE_8 + col;
        if (B_row < K && B_col < N)
            Bs_8[row][col] = B[B_row * N + B_col];
        else
            Bs_8[row][col] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE_8; ++k)
            acc += As_8[row][k] * Bs_8[k][col];

        __syncthreads();
    }

    int C_row = block_row * TILE_SIZE_8 + row;
    int C_col = block_col * TILE_SIZE_8 + col;
    if (C_row < M && C_col < N)
        C[C_row * N + C_col] = alpha * acc + beta * C[C_row * N + C_col];
}

// ========== TILE_SIZE = 16 ==========
#define TILE_SIZE_16 16
__shared__ float As_16[TILE_SIZE_16][TILE_SIZE_16];
__shared__ float Bs_16[TILE_SIZE_16][TILE_SIZE_16];

__global__ void shared_memory_gemm_kernel_16(float *C, const float *A, const float *B,
                                             int M, int N, int K,
                                             float alpha, float beta)
{
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    // 每个线程的累加器初始化
    float acc = 0.0f;
    // 根据k维度分块计算，每次加载一个tile到共享内存，循环次数为 ceil(K / 16)
    for (int tile = 0; tile < (K + TILE_SIZE_16 - 1) / TILE_SIZE_16; ++tile)
    {
        // 加载A和B的tile到共享内存
        // 16 个线程 row=0，col=0..15，bank = offset % 32 = (row * 16 + col) % 32 = (0 * 16 + col) % 32 = col % 32（0~15）
        // 16 个线程 row=1，col=0..15，bank = (16+col) % 32（16~31）
        int A_row = block_row * TILE_SIZE_16 + row;
        int A_col = tile * TILE_SIZE_16 + col;
        if (A_row < M && A_col < K)
            As_16[row][col] = A[A_row * K + A_col];
        else
            As_16[row][col] = 0.0f;

        int B_row = tile * TILE_SIZE_16 + row;
        int B_col = block_col * TILE_SIZE_16 + col;
        if (B_row < K && B_col < N)
            Bs_16[row][col] = B[B_row * N + B_col];
        else
            Bs_16[row][col] = 0.0f;
        // 等待所有线程加载完毕
        __syncthreads();

        // 同一个warp内的32个线程同时运行，row=0~1，col=0~15
        // 所有 row=0 的16个线程读As[0][k]，bank=k；所有row=1的16个线程读As[1][k]，bank=k+16，读相同地址，硬件广播，且两组bank不同，不冲突
        // 所有 row=0 和row=1的32个线程读 Bs[k][col]，bank = (k*16 + col) % 32；
        // 对于固定的 k：
        //   - 当 k 为偶数时，bank = col (0~15)
        //   - 当 k 为奇数时，bank = 16+col (16~31)
        // 不同的 row 但 col 相同的两个线程，读取的是同一个地址 Bs[k][col]，硬件广播，不冲突。
        for (int k = 0; k < TILE_SIZE_16; ++k)
            acc += As_16[row][k] * Bs_16[k][col];
        // 等待所有线程计算完毕，以免下一个tile覆盖共享内存
        __syncthreads();
    }
    // 计算结果写回全局内存
    int C_row = block_row * TILE_SIZE_16 + row;
    int C_col = block_col * TILE_SIZE_16 + col;
    if (C_row < M && C_col < N)
        C[C_row * N + C_col] = alpha * acc + beta * C[C_row * N + C_col];
}

// ========== TILE_SIZE = 32 ==========
#define TILE_SIZE_32 32
__shared__ float As_32[TILE_SIZE_32][TILE_SIZE_32];
__shared__ float Bs_32[TILE_SIZE_32][TILE_SIZE_32];

__global__ void shared_memory_gemm_kernel_32(float *C, const float *A, const float *B,
                                             int M, int N, int K,
                                             float alpha, float beta)
{
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    float acc = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE_32 - 1) / TILE_SIZE_32; ++tile)
    {
        int A_row = block_row * TILE_SIZE_32 + row;
        int A_col = tile * TILE_SIZE_32 + col;
        if (A_row < M && A_col < K)
            As_32[row][col] = A[A_row * K + A_col];
        else
            As_32[row][col] = 0.0f;

        int B_row = tile * TILE_SIZE_32 + row;
        int B_col = block_col * TILE_SIZE_32 + col;
        if (B_row < K && B_col < N)
            Bs_32[row][col] = B[B_row * N + B_col];
        else
            Bs_32[row][col] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE_32; ++k)
            acc += As_32[row][k] * Bs_32[k][col];

        __syncthreads();
    }

    int C_row = block_row * TILE_SIZE_32 + row;
    int C_col = block_col * TILE_SIZE_32 + col;
    if (C_row < M && C_col < N)
        C[C_row * N + C_col] = alpha * acc + beta * C[C_row * N + C_col];
}

// ========== TILE_SIZE = 32x8 ==========
#define TILE_M_32x8 8                               // blockDim.y
#define TILE_N_32x8 32                              // blockDim.x
#define TILE_K_32x8 32                              // tile size in K dimension
__shared__ float As_32x8[TILE_M_32x8][TILE_K_32x8]; // [8][32]
__shared__ float Bs_32x8[TILE_K_32x8][TILE_N_32x8]; // [32][32]

__global__ void shared_memory_gemm_kernel_32x8(float *C, const float *A, const float *B,
                                               int M, int N, int K,
                                               float alpha, float beta)
{
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int row = threadIdx.y; // 0..7
    int col = threadIdx.x; // 0..31

    float acc = 0.0f;

    for (int tile = 0; tile < (K + TILE_K_32x8 - 1) / TILE_K_32x8; ++tile)
    {
        // 加载 A tile: A[row, k_tile]
        int A_row = block_row * TILE_M_32x8 + row;
        int A_col = tile * TILE_K_32x8 + col;
        if (A_row < M && A_col < K)
            As_32x8[row][col] = A[A_row * K + A_col];
        else
            As_32x8[row][col] = 0.0f;

        // 加载 B tile: B[k_tile, col]
        int B_row = tile * TILE_K_32x8 + row;      // row 对应 K 维度
        int B_col = block_col * TILE_N_32x8 + col; // col 对应 N 维度
        if (B_row < K && B_col < N)
            Bs_32x8[row][col] = B[B_row * N + B_col];
        else
            Bs_32x8[row][col] = 0.0f;

        __syncthreads();

        // 内积：As[row][k] * Bs[k][col]
        for (int k = 0; k < TILE_K_32x8; ++k)
            acc += As_32x8[row][k] * Bs_32x8[k][col];

        __syncthreads();
    }

    int C_row = block_row * TILE_M_32x8 + row;
    int C_col = block_col * TILE_N_32x8 + col;
    if (C_row < M && C_col < N)
        C[C_row * N + C_col] = alpha * acc + beta * C[C_row * N + C_col];
}

// ========== TILE_SIZE = 8x32 ==========
#define TILE_M_8x32 32 // blockDim.y
#define TILE_N_8x32 8  // blockDim.x
#define TILE_K_8x32 32 // tile size in K dimension

__shared__ float As_8x32[TILE_M_8x32][TILE_K_8x32]; // [32][32]
__shared__ float Bs_8x32[TILE_K_8x32][TILE_N_8x32]; // [32][8]

__global__ void shared_memory_gemm_kernel_8x32(float *C, const float *A, const float *B,
                                               int M, int N, int K,
                                               float alpha, float beta)
{
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    float acc = 0.0f;

    for (int tile = 0; tile < (K + TILE_K_8x32 - 1) / TILE_K_8x32; ++tile)
    {
        int A_row = block_row * TILE_M_8x32 + row;
        int A_col = tile * TILE_K_8x32 + col;
        if (A_row < M && A_col < K)
            As_8x32[row][col] = A[A_row * K + A_col];
        else
            As_8x32[row][col] = 0.0f;

        int B_row = tile * TILE_K_8x32 + row;
        int B_col = block_col * TILE_N_8x32 + col;
        if (B_row < K && B_col < N)
            Bs_8x32[row][col] = B[B_row * N + B_col];
        else
            Bs_8x32[row][col] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_K_8x32; ++k)
            acc += As_8x32[row][k] * Bs_8x32[k][col];

        __syncthreads();
    }

    int C_row = block_row * TILE_M_8x32 + row;
    int C_col = block_col * TILE_N_8x32 + col;
    if (C_row < M && C_col < N)
        C[C_row * N + C_col] = alpha * acc + beta * C[C_row * N + C_col];
}

// ========== 启动函数 ==========
void run_shared_memory_kernel(float *d_C, const float *d_A, const float *d_B,
                              int M, int N, int K,
                              float alpha, float beta,
                              dim3 grid, dim3 block, cudaStream_t stream)
{
    if (block.x == 8 && block.y == 8)
    {
        shared_memory_gemm_kernel_8<<<grid, block, 0, stream>>>(d_C, d_A, d_B, M, N, K, alpha, beta);
    }
    else if (block.x == 16 && block.y == 16)
    {
        shared_memory_gemm_kernel_16<<<grid, block, 0, stream>>>(d_C, d_A, d_B, M, N, K, alpha, beta);
    }
    else if (block.x == 32 && block.y == 32)
    {
        shared_memory_gemm_kernel_32<<<grid, block, 0, stream>>>(d_C, d_A, d_B, M, N, K, alpha, beta);
    }
    else if (block.x == 32 && block.y == 8) // 32x8 tile
    {
        shared_memory_gemm_kernel_32x8<<<grid, block, 0, stream>>>(d_C, d_A, d_B, M, N, K, alpha, beta);
    }
    else if (block.x == 8 && block.y == 32) // 8x32 tile
    {
        shared_memory_gemm_kernel_8x32<<<grid, block, 0, stream>>>(d_C, d_A, d_B, M, N, K, alpha, beta);
    }
    else
    {
        printf("Warning: Unsupported block size (%d,%d), using default 16x16\n", block.x, block.y);
        shared_memory_gemm_kernel_16<<<grid, block, 0, stream>>>(d_C, d_A, d_B, M, N, K, alpha, beta);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}