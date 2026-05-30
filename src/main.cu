#include "gemm_common.h"
#include <iostream>
#include <string>
#include <cstring>

// 辅助函数：用随机数 [0,1) 初始化矩阵
void init_matrix(float *mat, int size)
{
    for (int i = 0; i < size; ++i)
    {
        mat[i] = (float)(rand() % 100) / 100.0f; // [0,1)
    }
}

// 计算 GFLOPS = 2*M*N*K / (1e9 *t)
double compute_gflops(int M, int N, int K, double time_ms)
{
    double flops = 2.0 * M * N * K;
    double time_sec = time_ms / 1000.0;
    return flops / (time_sec * 1e9);
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        printf("Usage: %s M N K kernel_type [block_x block_y]\n", argv[0]);
        printf("kernel_type: naive, shared, cublas, cpu\n");
        printf("Example: %s 1024 1024 1024 naive 16 16\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    std::string kernel_type = argv[4];

    int block_x = 16, block_y = 16;
    // 简单解析 block sizes
    if (argc >= 6)
        block_x = atoi(argv[5]);
    if (argc >= 7)
        block_y = atoi(argv[6]);

    const float alpha = 1.0f, beta = 0.0f;

    // 分配主机内存
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);
    float *h_C_ref = (float *)malloc(size_C);

    srand(12345);
    init_matrix(h_A, M * K);
    init_matrix(h_B, K * N);
    memset(h_C, 0, size_C);
    memset(h_C_ref, 0, size_C);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, size_A));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, size_B));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, size_C));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_C, 0, size_C));

    double elapsed_ms = 0.0;
    double gflops = 0.0;

    // 解析可选参数：支持 --validate 和 --ref=<cpu|cublas>
    bool validate = false;
    std::string ref_choice = "auto";
    for (int i = 5; i < argc; ++i)
    {
        if (strcmp(argv[i], "--validate") == 0)
            validate = true;
        else if (strncmp(argv[i], "--ref=", 6) == 0)
            ref_choice = std::string(argv[i] + 6);
    }
    // naive kernel (Stage 1)
    if (kernel_type == "naive")
    {
        dim3 block(block_x, block_y);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        printf("Running naive kernel with grid(%d,%d) block(%d,%d)\n",
               grid.x, grid.y, block.x, block.y);

        // Warmup
        run_naive_kernel(d_C, d_A, d_B, M, N, K, alpha, beta, grid, block);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Measure
        CudaTimer timer;
        timer.start();
        run_naive_kernel(d_C, d_A, d_B, M, N, K, alpha, beta, grid, block);
        timer.stop();
        elapsed_ms = timer.elapsed_ms();

        gflops = compute_gflops(M, N, K, elapsed_ms);
        printf("Time: %.3f ms, GFLOPS: %.2f\n", elapsed_ms, gflops);

        // Copy result back
        CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    }
    // Shared memory tiled kernel (Stage 2)
    else if (kernel_type == "shared")
    {
        dim3 block(block_x, block_y);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        printf("Running shared memory kernel with grid(%d,%d) block(%d,%d)\n",
               grid.x, grid.y, block.x, block.y);

        // Warmup
        run_shared_memory_kernel(d_C, d_A, d_B, M, N, K, alpha, beta, grid, block);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Measure
        CudaTimer timer;
        timer.start();
        run_shared_memory_kernel(d_C, d_A, d_B, M, N, K, alpha, beta, grid, block);
        timer.stop();
        elapsed_ms = timer.elapsed_ms();

        gflops = compute_gflops(M, N, K, elapsed_ms);
        printf("Time: %.3f ms, GFLOPS: %.2f\n", elapsed_ms, gflops);

        // Copy result back
        CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    }
    // Register blocking kernel (Stage 3)
    else if (kernel_type == "register")
    {
        int config = 1;
        for (int i = 5; i < argc; ++i)
        {
            if (strncmp(argv[i], "--config=", 9) == 0)
                config = atoi(argv[i] + 9);
        }
        printf("Running register blocking kernel config %d\n", config);

        // Warmup
        run_register_blocking_kernel(d_C, d_A, d_B, M, N, K, alpha, beta, 0, config);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Measure
        CudaTimer timer;
        timer.start();
        run_register_blocking_kernel(d_C, d_A, d_B, M, N, K, alpha, beta, 0, config);
        timer.stop();
        elapsed_ms = timer.elapsed_ms();

        gflops = compute_gflops(M, N, K, elapsed_ms);
        printf("Time: %.3f ms, GFLOPS: %.2f\n", elapsed_ms, gflops);

        // Copy result back
        CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    }
    // Bank conflict avoidance kernel (Stage 4) - 基于 Stage 3 的 Register Blocking + padding
    else if (kernel_type == "bank")
    {
        int config = 1;
        for (int i = 5; i < argc; ++i)
        {
            if (strncmp(argv[i], "--config=", 9) == 0)
                config = atoi(argv[i] + 9);
        }
        printf("Running bank conflict kernel (register blocking + padding) config %d\n", config);

        // Warmup
        run_bank_conflict_kernel(d_C, d_A, d_B, M, N, K, alpha, beta, 0, config);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Measure
        CudaTimer timer;
        timer.start();
        run_bank_conflict_kernel(d_C, d_A, d_B, M, N, K, alpha, beta, 0, config);
        timer.stop();
        elapsed_ms = timer.elapsed_ms();

        gflops = compute_gflops(M, N, K, elapsed_ms);
        printf("Time: %.3f ms, GFLOPS: %.2f\n", elapsed_ms, gflops);

        // Copy result back
        CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    }
    else if (kernel_type == "cublas")
    {
        cublasHandle_t handle;
        CHECK_CUBLAS_ERROR(cublasCreate(&handle));
        float ms;
        cublas_gemm(d_C, d_A, d_B, M, N, K, alpha, beta, handle, ms);
        elapsed_ms = ms;
        gflops = compute_gflops(M, N, K, elapsed_ms);
        printf("cuBLAS time: %.3f ms, GFLOPS: %.2f\n", elapsed_ms, gflops);
        CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
        cublasDestroy(handle);
    }
    else if (kernel_type == "cpu")
    {
        auto start = std::chrono::high_resolution_clock::now();
        cpu_gemm(h_C, h_A, h_B, M, N, K, alpha, beta);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        gflops = compute_gflops(M, N, K, elapsed_ms);
        printf("CPU time: %.3f ms, GFLOPS: %.2f\n", elapsed_ms, gflops);
    }
    else
    {
        printf("Unknown kernel type\n");
        return 1;
    }

    // 验证仅在请求时运行
    if (validate)
    {
        long long total_elements = static_cast<long long>(M) * static_cast<long long>(N);
        std::string ref_to_use = ref_choice;
        if (ref_to_use == "auto")
        {
            if (total_elements > 1048576) // > 1024*1024
                ref_to_use = "cublas";
            else
                ref_to_use = "cpu";
        }

        printf("Validation reference: %s (elements=%lld)\n", ref_to_use.c_str(), total_elements);

        // 计算参考结果
        if (ref_to_use == "cpu")
        {
            cpu_gemm(h_C_ref, h_A, h_B, M, N, K, alpha, beta);
        }
        else if (ref_to_use == "cublas")
        {
            cublasHandle_t ref_handle;
            CHECK_CUBLAS_ERROR(cublasCreate(&ref_handle));
            float ref_ms;
            float *d_C_ref = nullptr;
            CHECK_CUDA_ERROR(cudaMalloc(&d_C_ref, size_C));
            cublas_gemm(d_C_ref, d_A, d_B, M, N, K, alpha, beta, ref_handle, ref_ms);
            CHECK_CUDA_ERROR(cudaMemcpy(h_C_ref, d_C_ref, size_C, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaFree(d_C_ref));
            cublasDestroy(ref_handle);
        }

        float max_error = 0.0f;
        float avg_error = 0.0f;
        bool passed = validate_result(h_C, h_C_ref, M * N, 1e-5f, max_error, avg_error);

        printf("Max relative error: %e\n", max_error);
        printf("Avg relative error: %e\n", avg_error);
        printf("Validation passed: %d\n", passed ? 1 : 0);
    }
    else
        printf("Validation skipped\n");
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}