#pragma once
#include <atomic>
#include <string>
#include "rknn_api.h"
#include "yolov8.h"

struct NPUWorkerCtx {
    int id;
    int model_w = 640;
    int model_h = 640;
    rknn_core_mask core_mask;
    std::string model_path;
    rknn_app_context_t rknn_ctx;
    std::atomic<bool> running{true};
};

void worker_npu_func(NPUWorkerCtx& ctx, class NPUQueue& queue);
