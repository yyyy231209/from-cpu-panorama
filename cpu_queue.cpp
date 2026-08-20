#include "cpu_queue.h"

void CPUQueue::init()
{
    for (int i = 0; i < 4; i++) {
        SingleQueue& q = cam[i];
        q.head = 0;
        q.tail = 0;
        q.count = 0;
    }
}

bool CPUQueue::push(int cam_id, const cv::Mat& frame)
{
    if (cam_id < 0 || cam_id >= 4) return false;
    SingleQueue& q = cam[cam_id];

    std::unique_lock<std::mutex> lk(q.mtx);
    if (q.count >= CPU_QUEUE_SIZE) {
        q.head = (q.head + 1) % CPU_QUEUE_SIZE;
        q.count--;
    }
    int idx = q.tail;
    q.tail = (q.tail + 1) % CPU_QUEUE_SIZE;
    q.count++;
    lk.unlock();

    q.buf[idx] = frame.clone();

    q.cv.notify_one();
    return true;
}

bool CPUQueue::pop_all(cv::Mat frames[4])
{
    for (int i = 0; i < 4; i++) {
        SingleQueue& q = cam[i];
        std::unique_lock<std::mutex> lk(q.mtx);
        if (q.count == 0) return false;
        int idx = q.head;
        q.head = (q.head + 1) % CPU_QUEUE_SIZE;
        q.count--;
        frames[i] = q.buf[idx].clone();
    }
    return true;
}

void CPUQueue::free_bufs()
{
    for (int i = 0; i < 4; i++) {
        SingleQueue& q = cam[i];
        std::lock_guard<std::mutex> lk(q.mtx);
        for (int j = 0; j < CPU_QUEUE_SIZE; j++) {
            q.buf[j].release();
        }
    }
}
