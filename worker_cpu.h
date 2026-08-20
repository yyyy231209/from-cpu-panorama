#pragma once
#include <string>
#include <atomic>
#include <opencv2/opencv.hpp>

struct CPUWorkerCtx {
    int id;
    std::string dev_path;
    int cam_fd;
    std::atomic<bool> running{true};
};

void worker_cpu_func(CPUWorkerCtx& ctx, class CPUQueue& queue);
