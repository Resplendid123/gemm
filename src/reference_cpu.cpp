#include "gemm_common.h"

// CPU 矩阵乘法 baseline
// C = alpha * A * B + beta * C （用于控制缩放）
// A (M×K)  ×  B (K×N)  =  C (M×N)
// A[i][k] 索引：i*K + k
// B[k][j] 索引：k*N + j
// C[i][j] 索引：i*N + j
void cpu_gemm(float* C, const float* A, const float* B,
              int M, int N, int K,
              float alpha, float beta) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}