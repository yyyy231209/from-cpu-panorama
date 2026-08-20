#include "cpu_warp.h"
#include "cpu_queue.h"
#include "npu_queue.h"
#include "cpu_H_fuc.h"
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>

void cpu_warp_func(WarpThreadCtx& wctx, CPUQueue& cpu_queue, NPUQueue& npu_queue) {
    cv::Mat frames[4];
    cv::Mat stitched;
    auto t0 = std::chrono::steady_clock::now();
    int fcount = 0;

    printf("[stitch] thread started\n");

    while (wctx.running.load()) {
        bool all_ok = true;
        for (int i = 0; i < 4; i++) {
            SingleQueue& q = cpu_queue.cam[i];
            std::unique_lock<std::mutex> lk(q.mtx);
            if (q.count == 0)
                q.cv.wait_for(lk, std::chrono::milliseconds(10));
            if (q.count == 0) { all_ok = false; break; }
            int idx = q.head;
            q.head = (q.head + 1) % CPU_QUEUE_SIZE;
            q.count--;
            frames[i] = q.buf[idx].clone();
        }
        if (!all_ok) continue;

        stitch_four_cpu(frames[0], frames[1], frames[2], frames[3],
                        wctx.maps, stitched);

        if (stitched.empty()) continue;

        int s_cw = stitched.cols;
        int s_ch = stitched.rows;
        int square = std::max(s_cw, s_ch);
        square = ((square + 15) / 16) * 16;
        cv::Mat padded(square, square, CV_8UC3, cv::Scalar(0, 0, 0));
        int off_x = (square - s_cw) / 2;
        int off_y = (square - s_ch) / 2;
        if (off_x < 0) off_x = 0;
        if (off_y < 0) off_y = 0;
        stitched.copyTo(padded(cv::Rect(off_x, off_y, s_cw, s_ch)));

        int npu_idx = npu_queue.claim_free();
        if (npu_idx < 0) {
            std::this_thread::yield();
            continue;
        }

        rga_buffer_t pad_rga = wrapbuffer_virtualaddr(padded.data, square, square,
            RK_FORMAT_BGR_888, square, square);

        IM_STATUS status = imresize(pad_rga, npu_queue.rga_buf[npu_idx]);
        if (status != IM_STATUS_SUCCESS) {
            printf("[stitch] RGA resize fail: %s\n", imStrError((IM_STATUS)status));
            npu_queue.mark_free(npu_idx);
            continue;
        }

        wctx.frame_id++;
        npu_queue.mark_filled(npu_idx, wctx.frame_id);

        fcount++;
        auto t1 = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        if (ms >= 2000) {
            printf("[stitch] FPS: %.1f\n", fcount * 1000.0 / ms);
            fcount = 0;
            t0 = t1;
        }
    }

    printf("[stitch] stopped\n");
}
