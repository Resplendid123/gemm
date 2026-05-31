#include "gemm_common.h"
#include <cmath>
#include <cstdio>

// 验证两个浮点数组是否相等，并返回最大相对误差
bool validate_result(const float *gpu_result, const float *ref_result,
                     int size, float eps, float &max_error, float &avg_error)
{
    max_error = 0.0f;
    double sum_rel = 0.0;
    int mismatch_count = 0;

    for (int i = 0; i < size; ++i)
    {
        float diff = fabs(gpu_result[i] - ref_result[i]);
        float rel = diff / (fabs(ref_result[i]) + 1e-8f);

        if (rel > max_error)
            max_error = rel;

        sum_rel += rel;

        if (rel > eps)
            mismatch_count++;
    }

    avg_error = static_cast<float>(sum_rel / size);

    if (mismatch_count > 0)
    {
        printf("发现 %d 个元素不匹配 (tolerance: %e)\n", mismatch_count, eps);
        printf("最大相对误差: %e, 平均相对误差: %e\n", max_error, avg_error);
        return false;
    }

    return true;
}