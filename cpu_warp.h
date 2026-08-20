#pragma once
#include <opencv2/opencv.hpp>
#include <atomic>
#include "cpu_H_fuc.h"
#include "rga/im2d.h"

struct WarpThreadCtx {
    cv::Mat H12, H32, H42;
    cv::Mat T;
    StitchMaps maps;
    int canvas_w = 0;
    int canvas_h = 0;
    int model_w = 1024;
    int model_h = 1024;
    std::atomic<bool> running{true};
    int frame_id = 0;
};

void cpu_warp_func(WarpThreadCtx& wctx, class CPUQueue& cpu_queue, class NPUQueue& npu_queue);
