#include "gemm_common.h"
#include <cmath>
#include <cstdio>

// 验证两个浮点数组是否相等，并返回最大相对误差
bool validate_result(const float *gpu_result, const float *ref_result,
                     int size, float eps, float &max_error, float &avg_error)
{
    max_error = 0.0f;
    double sum_rel = 0.0;
    for (int i = 0; i < size; ++i)
    {
        float diff = fabs(gpu_result[i] - ref_result[i]);
        float rel = diff / (fabs(ref_result[i]) + 1e-8f);

        if (rel > max_error)
        {
            max_error = rel;
        }

        sum_rel += rel;

        if (rel > eps)
        {
            printf("元素 %d 不匹配: GPU=%f, 参考=%f, 绝对误差=%f, 相对误差=%e\n",
                   i, gpu_result[i], ref_result[i], diff, rel);
            avg_error = static_cast<float>(sum_rel / (i + 1));
            return false;
        }
    }

    avg_error = static_cast<float>(sum_rel / size);
    return true;
}