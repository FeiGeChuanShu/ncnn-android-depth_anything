#ifndef DPT_H
#define DPT_H

#include <opencv2/core/core.hpp>
#include <net.h>

class Dpt
{
public:
    Dpt();

    int load(const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int load(AAssetManager* mgr, const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int detect(const cv::Mat& rgb, cv::Mat& depth_color);

    int draw(cv::Mat& rgb, cv::Mat& depth_color);

private:
    ncnn::Net dpt_;
    int target_size_;
    float mean_vals_[3];
    float norm_vals_[3];
    cv::Mat color_map_;
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // DPT_H
