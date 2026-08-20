#include "npu_queue.h"
#include <cstdio>

void NPUQueue::init(int model_w, int model_h)
{
    size_t buf_size = (size_t)model_w * model_h * 3;
    size_t alloc_size = buf_size + sizeof(NPUBufferMeta);

    for (int i = 0; i < buf_num; i++) {
        dma_buf_alloc(DMA_HEAP_PATH, alloc_size, &fd[i], &virt[i]);
        rga_buf[i] = wrapbuffer_fd(fd[i], model_w, model_h, RK_FORMAT_RGB_888, model_w, model_h);
        cv_buf[i] = cv::Mat(model_h, model_w, CV_8UC3, virt[i]);
        meta[i] = (NPUBufferMeta*)((char*)virt[i] + buf_size);
        meta[i]->frame_id = -1;
        meta[i]->state = NPU_BUF_FREE;
    }

    dma_buf_alloc(DMA_HEAP_PATH, alloc_size, &display_fd, &display_virt);
    display_rga = wrapbuffer_fd(display_fd, model_w, model_h, RK_FORMAT_RGB_888, model_w, model_h);
    display_cv = cv::Mat(model_h, model_w, CV_8UC3, display_virt);

    printf("[npu_queue] %d buffers allocated, size=%dx%d\n", buf_num, model_w, model_h);
}

int NPUQueue::claim_free()
{
    std::lock_guard<std::mutex> lk(mtx);
    for (int i = 0; i < buf_num; i++) {
        if (meta[i]->state == NPU_BUF_FREE) {
            meta[i]->state = NPU_BUF_PROCESSING;
            return i;
        }
    }
    return -1;
}

void NPUQueue::mark_filled(int idx, int frame_id)
{
    std::lock_guard<std::mutex> lk(mtx);
    if (idx < 0 || idx >= buf_num) return;
    meta[idx]->frame_id = frame_id;
    meta[idx]->state = NPU_BUF_FILLED;
}

int NPUQueue::claim_filled()
{
    std::lock_guard<std::mutex> lk(mtx);
    for (int i = 0; i < buf_num; i++) {
        if (meta[i]->state == NPU_BUF_FILLED) {
            meta[i]->state = NPU_BUF_PROCESSING;
            return i;
        }
    }
    return -1;
}

void NPUQueue::mark_ready(int idx)
{
    std::lock_guard<std::mutex> lk(mtx);
    if (idx < 0 || idx >= buf_num) return;
    meta[idx]->state = NPU_BUF_READY;
}

int NPUQueue::claim_ready()
{
    std::lock_guard<std::mutex> lk(mtx);
    int best_idx = -1;
    int best_id = -1;
    for (int i = 0; i < buf_num; i++) {
        if (meta[i]->state == NPU_BUF_READY) {
            if (best_idx == -1 || meta[i]->frame_id < best_id) {
                best_idx = i;
                best_id = meta[i]->frame_id;
            }
        }
    }
    if (best_idx >= 0) {
        meta[best_idx]->state = NPU_BUF_PROCESSING;
    }
    return best_idx;
}

void NPUQueue::mark_free(int idx)
{
    std::lock_guard<std::mutex> lk(mtx);
    if (idx < 0 || idx >= buf_num) return;
    meta[idx]->frame_id = -1;
    meta[idx]->state = NPU_BUF_FREE;
}

void NPUQueue::destroy()
{
    size_t buf_size = 0;
    if (buf_num > 0 && cv_buf[0].data) {
        buf_size = (size_t)cv_buf[0].cols * cv_buf[0].rows * 3;
    }
    size_t alloc_size = buf_size + sizeof(NPUBufferMeta);
    for (int i = 0; i < buf_num; i++) {
        if (fd[i] >= 0) dma_buf_free(alloc_size, &fd[i], virt[i]);
    }
    if (display_fd >= 0) dma_buf_free(alloc_size, &display_fd, display_virt);
}
