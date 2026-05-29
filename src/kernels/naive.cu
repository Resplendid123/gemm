#include "gemm_common.h"

// 朴素 CUDA 内核：每个线程计算输出 C 的一个元素
__global__ void naive_gemm_kernel(float* C, const float* A, const float* B,
                                  int M, int N, int K,
                                  float alpha, float beta) {
    // 计算当前线程负责的输出矩阵的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：防止线程超出矩阵范围（处理非对齐尺寸）
    if (row < M && col < N) {
        float sum = 0.0f;
        // 沿 K 维度累加
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// 启动朴素内核的封装函数
void run_naive_kernel(float* d_C, const float* d_A, const float* d_B,
                      int M, int N, int K,
                      float alpha, float beta,
                      dim3 grid, dim3 block, cudaStream_t stream) {
    naive_gemm_kernel<<<grid, block, 0, stream>>>(d_C, d_A, d_B, M, N, K, alpha, beta);
    CHECK_CUDA_ERROR(cudaGetLastError());
}