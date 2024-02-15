#include "dpt.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cpu.h"


Dpt::Dpt()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}


int Dpt::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    dpt_.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    dpt_.opt = ncnn::Option();

#if NCNN_VULKAN
    dpt_.opt.use_vulkan_compute = use_gpu;
#endif

    dpt_.opt.num_threads = ncnn::get_big_cpu_count();
    dpt_.opt.blob_allocator = &blob_pool_allocator;
    dpt_.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "dpt%s.param", modeltype);
    sprintf(modelpath, "dpt%s.bin", modeltype);

    dpt_.load_param(mgr, parampath);
    dpt_.load_model(mgr, modelpath);

    target_size_ = _target_size;
    mean_vals_[0] = _mean_vals[0];
    mean_vals_[1] = _mean_vals[1];
    mean_vals_[2] = _mean_vals[2];
    norm_vals_[0] = _norm_vals[0];
    norm_vals_[1] = _norm_vals[1];
    norm_vals_[2] = _norm_vals[2];
    
    color_map_ = cv::Mat(target_size_, target_size_, CV_8UC3);

    return 0;
}

int Dpt::detect(const cv::Mat& rgb, cv::Mat& depth_color)
{
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size_ / w;
        w = target_size_;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size_ / h;
        h = target_size_;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);

    // pad to target_size rectangle
    int wpad = target_size_ - w;
    int hpad = target_size_ - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(mean_vals_, norm_vals_);

    ncnn::Extractor ex = dpt_.create_extractor();

    ex.input("image", in_pad);

    ncnn::Mat out;
    ex.extract("depth", out);

    cv::Mat depth(out.h, out.w, CV_32FC1, (void*)out.data);
    cv::normalize(depth, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::applyColorMap(depth, color_map_, cv::ColormapTypes::COLORMAP_INFERNO);
    cv::resize(color_map_(cv::Rect(wpad / 2, hpad / 2, w, h)), depth_color, rgb.size());

    return 0;
}

int Dpt::draw(cv::Mat& rgb, cv::Mat& depth_color)
{
    cv::cvtColor(depth_color, rgb, cv::COLOR_RGB2BGR);

    return 0;
}
