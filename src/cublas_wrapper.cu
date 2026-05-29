#include "gemm_common.h"

// cublas 矩阵乘法 baseline，提供行主序接口
void cublas_gemm(float *C, const float *A, const float *B,
                 int M, int N, int K,
                 float alpha, float beta,
                 cublasHandle_t handle, float &elapsed_ms)
{
    // 行主序 C(M,N) = alpha * A(M,K) * B(K,N) + beta*C
    // 行主序传入 cublas 时，矩阵已经被"隐式转置"了，对应的是 A^T(K,M)*B^T(N,K)，因此需要交换 A/B 的顺序，并使用非转置操作符

    const size_t size_C = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
    float *d_warmup = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_warmup, size_C));
    CHECK_CUDA_ERROR(cudaMemset(d_warmup, 0, size_C));

    // 先做一次不计时的 warm-up，避开 cuBLAS 的首次初始化和算法选择开销
    CHECK_CUBLAS_ERROR(cublasSgemm(handle,
                                   CUBLAS_OP_N, CUBLAS_OP_N,
                                   N, M, K,
                                   &alpha,
                                   B, N,
                                   A, K,
                                   &beta,
                                   d_warmup, N));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.start();

    // cublasSgemm 参数说明：
    // handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc
    CHECK_CUBLAS_ERROR(cublasSgemm(handle,                   // 上下文句柄
                                   CUBLAS_OP_N, CUBLAS_OP_N, // 对矩阵A、B不做转置
                                   N, M, K,                  // 结果矩阵按列主序的维度
                                   &alpha,                   // 缩放因子α
                                   B, N,                     // 列主序下的第一项 B^T(N,K)
                                   A, K,                     // 列主序下的第二项 A^T(K,M)
                                   &beta,                    // 缩放因子β
                                   C, N));                   // 结果矩阵 C^T(N,M)

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    timer.stop();
    elapsed_ms = timer.elapsed_ms();

    CHECK_CUDA_ERROR(cudaFree(d_warmup));
}