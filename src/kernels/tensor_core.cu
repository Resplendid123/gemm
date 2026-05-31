#include "gemm_common.h"
#include <mma.h>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// ========== 设备端类型转换 Kernels ==========
__global__ void float2half_kernel(half *dst, const float *src, size_t n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        dst[idx] = __float2half(src[idx]);
}

__global__ void half2float_kernel(float *dst, const half *src, size_t n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        dst[idx] = __half2float(src[idx]);
}

// ========== 配置1: FP16 输出（双缓冲） ==========
template <int BM, int BN>
__global__ void wmma_gemm_fp16_db(half *C, const half *A, const half *B,
                                  int M, int N, int K,
                                  float alpha, float beta)
{
    constexpr int BK = WMMA_K;
    constexpr int WN = BN / WMMA_N; // 每个block中warp的列数

    // 双缓冲共享内存
    __shared__ half As[2][BM][BK];
    __shared__ half Bs[2][BN][BK];
    // 整个 block 的输出缓冲区
    __shared__ float tmp[BM][BN];

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_m = warp_id / WN; // 当前warp负责的A行块索引
    int warp_n = warp_id % WN; // 当前warp负责的B列块索引

    // 当前 warp 在共享内存 tmp 中的起始偏移
    int warp_tile_row = warp_m * WMMA_M;
    int warp_tile_col = warp_n * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int numTiles = (K + BK - 1) / BK;
    int cur_buffer = 0;

    // ----- 预取第一个tile到 cur_buffer -----
    int k0 = 0;
    // 加载A (行主序)
    for (int i = tid; i < BM * BK; i += blockDim.x)
    {
        int row = i / BK;
        int col = i % BK;
        int g_row = block_row + row;
        int g_col = k0 + col;
        if (g_row < M && g_col < K)
            As[cur_buffer][row][col] = A[g_row * K + g_col];
        else
            As[cur_buffer][row][col] = __float2half(0.0f);
    }
    // 加载B (列主序)
    for (int i = tid; i < BK * BN; i += blockDim.x)
    {
        int col = i / BK; // 列索引 (第一维)
        int row = i % BK; // 行索引 (第二维)
        int g_row = k0 + row;
        int g_col = block_col + col;
        if (g_row < K && g_col < N)
            Bs[cur_buffer][col][row] = B[g_row * N + g_col];
        else
            Bs[cur_buffer][col][row] = __float2half(0.0f);
    }
    __syncthreads();

    // ----- 主循环 -----
    for (int tile = 0; tile < numTiles; ++tile)
    {
        // 计算当前缓冲区
        wmma::load_matrix_sync(a_frag, As[cur_buffer][warp_m * WMMA_M], BK);
        wmma::load_matrix_sync(b_frag, &Bs[cur_buffer][warp_n * WMMA_N][0], BK);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        int next_buffer = cur_buffer ^ 1;
        if (tile + 1 < numTiles)
        {
            int k_next = (tile + 1) * BK;
            // 异步加载下一个A tile
            for (int i = tid; i < BM * BK; i += blockDim.x)
            {
                int row = i / BK;
                int col = i % BK;
                int g_row = block_row + row;
                int g_col = k_next + col;
                if (g_row < M && g_col < K)
                    As[next_buffer][row][col] = A[g_row * K + g_col];
                else
                    As[next_buffer][row][col] = __float2half(0.0f);
            }
            // 异步加载下一个B tile (列主序)
            for (int i = tid; i < BK * BN; i += blockDim.x)
            {
                int col = i / BK;
                int row = i % BK;
                int g_row = k_next + row;
                int g_col = block_col + col;
                if (g_row < K && g_col < N)
                    Bs[next_buffer][col][row] = B[g_row * N + g_col];
                else
                    Bs[next_buffer][col][row] = __float2half(0.0f);
            }
        }
        __syncthreads();
        cur_buffer = next_buffer;
    }

    // 将 fragment 存储到共享内存 tmp 中的对应子区域
    wmma::store_matrix_sync(&tmp[warp_tile_row][warp_tile_col], c_frag, BN, wmma::mem_row_major);
    __syncthreads();

    // 每个线程协作写回整个 block 的输出 (每个线程写 8 个元素)
    for (int idx = lane_id; idx < WMMA_M * WMMA_N; idx += 32)
    {
        int row = idx / WMMA_N;
        int col = idx % WMMA_N;
        int global_row = block_row + warp_tile_row + row;
        int global_col = block_col + warp_tile_col + col;
        if (global_row < M && global_col < N)
        {
            float val = tmp[warp_tile_row + row][warp_tile_col + col];
            int C_idx = global_row * N + global_col;
            half old = C[C_idx];
            float res = alpha * val + beta * __half2float(old);
            C[C_idx] = __float2half(res);
        }
    }
}

// ========== 配置2: FP32 输出（双缓冲） ==========
template <int BM, int BN>
__global__ void wmma_gemm_fp32(float *C, const half *A, const half *B,
                               int M, int N, int K,
                               float alpha, float beta)
{
    constexpr int BK = WMMA_K;
    constexpr int WN = BN / WMMA_N;

    __shared__ half As[2][BM][BK];
    __shared__ half Bs[2][BN][BK];
    __shared__ float tmp[BM][BN]; // 整个 block 的输出缓冲区

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_m = warp_id / WN;
    int warp_n = warp_id % WN;

    int warp_tile_row = warp_m * WMMA_M;
    int warp_tile_col = warp_n * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int numTiles = (K + BK - 1) / BK;
    int cur_buffer = 0;

    // 预取 tile0
    int k0 = 0;
    for (int i = tid; i < BM * BK; i += blockDim.x)
    {
        int row = i / BK, col = i % BK;
        int g_row = block_row + row, g_col = k0 + col;
        if (g_row < M && g_col < K)
            As[cur_buffer][row][col] = A[g_row * K + g_col];
        else
            As[cur_buffer][row][col] = __float2half(0.0f);
    }
    for (int i = tid; i < BK * BN; i += blockDim.x)
    {
        int col = i / BK, row = i % BK;
        int g_row = k0 + row, g_col = block_col + col;
        if (g_row < K && g_col < N)
            Bs[cur_buffer][col][row] = B[g_row * N + g_col];
        else
            Bs[cur_buffer][col][row] = __float2half(0.0f);
    }
    __syncthreads();

    for (int tile = 0; tile < numTiles; ++tile)
    {
        wmma::load_matrix_sync(a_frag, As[cur_buffer][warp_m * WMMA_M], BK);
        wmma::load_matrix_sync(b_frag, &Bs[cur_buffer][warp_n * WMMA_N][0], BK);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        int next_buffer = cur_buffer ^ 1;
        if (tile + 1 < numTiles)
        {
            int k_next = (tile + 1) * BK;
            for (int i = tid; i < BM * BK; i += blockDim.x)
            {
                int row = i / BK, col = i % BK;
                int g_row = block_row + row, g_col = k_next + col;
                if (g_row < M && g_col < K)
                    As[next_buffer][row][col] = A[g_row * K + g_col];
                else
                    As[next_buffer][row][col] = __float2half(0.0f);
            }
            for (int i = tid; i < BK * BN; i += blockDim.x)
            {
                int col = i / BK, row = i % BK;
                int g_row = k_next + row, g_col = block_col + col;
                if (g_row < K && g_col < N)
                    Bs[next_buffer][col][row] = B[g_row * N + g_col];
                else
                    Bs[next_buffer][col][row] = __float2half(0.0f);
            }
        }
        __syncthreads();
        cur_buffer = next_buffer;
    }

    // 存储到共享内存 tmp 的对应子区域
    wmma::store_matrix_sync(&tmp[warp_tile_row][warp_tile_col], c_frag, BN, wmma::mem_row_major);
    __syncthreads();

    // 每个线程协作写回全局内存
    for (int idx = lane_id; idx < WMMA_M * WMMA_N; idx += 32)
    {
        int row = idx / WMMA_N;
        int col = idx % WMMA_N;
        int global_row = block_row + warp_tile_row + row;
        int global_col = block_col + warp_tile_col + col;
        if (global_row < M && global_col < N)
        {
            float val = tmp[warp_tile_row + row][warp_tile_col + col];
            C[global_row * N + global_col] = alpha * val + beta * C[global_row * N + global_col];
        }
    }
}

// ========== 启动函数 ==========
void run_tensor_core_kernel(float *d_C, const float *d_A, const float *d_B,
                            int M, int N, int K,
                            float alpha, float beta,
                            cudaStream_t stream, int config)
{
    // 分配 half 输入缓冲区
    half *d_A_half = nullptr;
    half *d_B_half = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A_half, M * K * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B_half, K * N * sizeof(half)));

    // 设备端 float -> half 转换
    int block_conv = 256;
    int grid_a = (M * K + block_conv - 1) / block_conv;
    int grid_b = (K * N + block_conv - 1) / block_conv;
    float2half_kernel<<<grid_a, block_conv, 0, stream>>>(d_A_half, d_A, M * K);
    float2half_kernel<<<grid_b, block_conv, 0, stream>>>(d_B_half, d_B, K * N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    constexpr int BM = 64, BN = 64;
    constexpr int WM = BM / WMMA_M;
    constexpr int WN = BN / WMMA_N;
    constexpr int NUM_WARPS = WM * WN;

    int grid_x = (N + BN - 1) / BN;
    int grid_y = (M + BM - 1) / BM;
    dim3 grid(grid_x, grid_y);
    dim3 block(NUM_WARPS * 32);

    if (config == 1)
    {
        // FP16 输出
        half *d_C_half = nullptr;
        CHECK_CUDA_ERROR(cudaMalloc(&d_C_half, M * N * sizeof(half)));
        CHECK_CUDA_ERROR(cudaMemset(d_C_half, 0, M * N * sizeof(half)));

        wmma_gemm_fp16_db<BM, BN><<<grid, block, 0, stream>>>(
            d_C_half, d_A_half, d_B_half, M, N, K, alpha, beta);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // 设备端 half -> float 转换
        int grid_c = (M * N + block_conv - 1) / block_conv;
        half2float_kernel<<<grid_c, block_conv, 0, stream>>>(d_C, d_C_half, M * N);
        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaFree(d_C_half));
    }
    else
    {
        // FP32 输出
        wmma_gemm_fp32<BM, BN><<<grid, block, 0, stream>>>(
            d_C, d_A_half, d_B_half, M, N, K, alpha, beta);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    CHECK_CUDA_ERROR(cudaFree(d_A_half));
    CHECK_CUDA_ERROR(cudaFree(d_B_half));
}