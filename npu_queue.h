#pragma once
#include "rga/RgaApi.h"
#include "rga/im2d.h"
#include <mutex>
#include "dma_alloc.h"
#include <opencv2/opencv.hpp>

#define DMA_HEAP_PATH "/dev/dma_heap/system-uncached"

constexpr int NPU_QUEUE_CAPACITY = 6;

enum NPUBufState {
    NPU_BUF_FREE = 0,
    NPU_BUF_FILLED = 1,
    NPU_BUF_PROCESSING = 2,
    NPU_BUF_READY = 3
};

struct NPUBufferMeta {
    int frame_id = -1;
    int state = NPU_BUF_FREE;
};

struct NPUQueue {
    int buf_num = NPU_QUEUE_CAPACITY;
    std::mutex mtx;

    int fd[NPU_QUEUE_CAPACITY];
    void* virt[NPU_QUEUE_CAPACITY];

    rga_buffer_t rga_buf[NPU_QUEUE_CAPACITY];
    cv::Mat cv_buf[NPU_QUEUE_CAPACITY];

    NPUBufferMeta* meta[NPU_QUEUE_CAPACITY];

    int display_fd = -1;
    void* display_virt = nullptr;
    rga_buffer_t display_rga;
    cv::Mat display_cv;

    void init(int model_w, int model_h);
    int claim_free();
    void mark_filled(int idx, int frame_id);
    int claim_filled();
    void mark_ready(int idx);
    int claim_ready();
    void mark_free(int idx);
    void destroy();
};
