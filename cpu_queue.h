#pragma once
#include <opencv2/opencv.hpp>
#include <mutex>
#include <condition_variable>

constexpr int CPU_QUEUE_SIZE = 4;

struct SingleQueue {
    cv::Mat buf[CPU_QUEUE_SIZE];
    int head = 0;
    int tail = 0;
    int count = 0;
    std::mutex mtx;
    std::condition_variable cv;
};

struct CPUQueue {
    SingleQueue cam[4];

    void init();
    bool push(int cam_id, const cv::Mat& frame);
    bool pop_all(cv::Mat frames[4]);
    void free_bufs();
};
