#pragma once
#include <opencv2/opencv.hpp>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>

class MjpegStreamer {
public:
    MjpegStreamer(int port = 8080);
    ~MjpegStreamer();

    void start();
    void stop();
    void push_frame(const cv::Mat& frame);

private:
    int port_;
    int server_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread server_thread_;

    std::vector<uchar> jpeg_buf_;
    std::mutex jpeg_mtx_;
    std::condition_variable jpeg_cv_;
    bool new_frame_ = false;

    void server_loop();
};
