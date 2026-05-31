#ifndef GEMM_COMMON_H
#define GEMM_COMMON_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

// 宏：检查 CUDA 运行时 API 调用是否出错
#define CHECK_CUDA_ERROR(call)                                               \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// 宏：检查 cuBLAS API 调用是否出错
#define CHECK_CUBLAS_ERROR(call)                                            \
    do                                                                      \
    {                                                                       \
        cublasStatus_t stat = call;                                         \
        if (stat != CUBLAS_STATUS_SUCCESS)                                  \
        {                                                                   \
            fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ========== 函数声明 ==========
// CPU 参考 GEMM，行主序，支持 alpha 和 beta
void cpu_gemm(float *C, const float *A, const float *B,
              int M, int N, int K,
              float alpha, float beta);

// cuBLAS 封装函数：以行主序接口调用 cublasSgemm
void cublas_gemm(float *C, const float *A, const float *B,
                 int M, int N, int K,
                 float alpha, float beta,
                 cublasHandle_t handle, float &elapsed_ms);

// Naive CUDA kernel 启动封装
void run_naive_kernel(float *d_C, const float *d_A, const float *d_B,
                      int M, int N, int K,
                      float alpha, float beta,
                      dim3 grid, dim3 block, cudaStream_t stream = 0);

// Shared Memory Tiled CUDA kernel 启动封装
void run_shared_memory_kernel(float *d_C, const float *d_A, const float *d_B,
                              int M, int N, int K,
                              float alpha, float beta,
                              cudaStream_t stream, int config);

// Register Blocking GEMM kernel 启动封装
void run_register_blocking_kernel(float *d_C, const float *d_A, const float *d_B,
                                  int M, int N, int K,
                                  float alpha, float beta,
                                  cudaStream_t stream,
                                  int config);

// Bank Conflict Avoidance GEMM kernel 启动封装
void run_bank_conflict_kernel(float *d_C, const float *d_A, const float *d_B,
                              int M, int N, int K,
                              float alpha, float beta,
                              cudaStream_t stream, int config);

// Double Buffering GEMM kernel 启动封装
void run_double_buffer_kernel(float *d_C, const float *d_A, const float *d_B,
                              int M, int N, int K,
                              float alpha, float beta,
                              cudaStream_t stream, int config);

// Tensor Core GEMM kernel 启动封装
void run_tensor_core_kernel(float *d_C, const float *d_A, const float *d_B,
                            int M, int N, int K,
                            float alpha, float beta,
                            cudaStream_t stream, int config);

// 验证 GPU 计算结果与参考结果是否一致（允许相对误差）
bool validate_result(const float *gpu_result, const float *ref_result,
                     int size, float eps, float &max_error, float &avg_error);

// ========== 计时器类 ==========
// 使用 CUDA Event 的计时器类，用于精确测量 GPU 内核执行时间
class CudaTimer
{
public:
    CudaTimer()
    {
        CHECK_CUDA_ERROR(cudaEventCreate(&start_));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop_));
    }
    ~CudaTimer()
    {
        CHECK_CUDA_ERROR(cudaEventDestroy(start_));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop_));
    }
    void start() { CHECK_CUDA_ERROR(cudaEventRecord(start_)); }
    void stop() { CHECK_CUDA_ERROR(cudaEventRecord(stop_)); }
    // 计算两个事件间的毫秒数
    float elapsed_ms()
    {
        float ms;
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop_));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

#endif // GEMM_COMMON_H