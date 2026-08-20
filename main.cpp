#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <thread>
#include <chrono>
#include <vector>
#include <string>

#include "cpu_queue.h"
#include "npu_queue.h"
#include "cpu_H_fuc.h"
#include "cpu_warp.h"
#include "worker_cpu.h"
#include "worker_npu.h"

#include "rga/im2d.h"
#include "dma_alloc.h"
#include "mjpeg_streamer.h"

#define NUM_NPU 3

static std::atomic<bool> g_exit{false};

void signal_handler(int) { g_exit.store(true); }

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    std::string h_dir = ".";
    std::string model_path = "best.rknn";
    const char* cam_devs[4] = {
        "/dev/video21", "/dev/video23", "/dev/video25", "/dev/video27"
    };

    if (argc > 1) h_dir = argv[1];
    if (argc > 2) model_path = argv[2];

    // ---- load H matrices ----
    WarpThreadCtx wctx;
    if (!load_H_matrices(h_dir, wctx.H12, wctx.H32, wctx.H42,
                         wctx.canvas_w, wctx.canvas_h, wctx.T, wctx.maps)) {
        printf("ERROR: failed to load H matrices from %s\n", h_dir.c_str());
        printf("Place H12.bin H32.bin H42.bin in the directory\n");
        return -1;
    }
    printf("Canvas size: %dx%d\n", wctx.canvas_w, wctx.canvas_h);

    // ---- init NPU queue (model is 1024x1024) ----
    NPUQueue npu_queue;
    int model_w = 1024, model_h = 1024;
    npu_queue.init(model_w, model_h);

    // ---- init CPU queue ----
    CPUQueue cpu_queue;
    cpu_queue.init();

    // ---- start V4L2 capture threads ----
    std::vector<CPUWorkerCtx> cpu_ctxs(4);
    std::vector<std::thread> cpu_threads;
    for (int i = 0; i < 4; i++) {
        cpu_ctxs[i].id = i;
        cpu_ctxs[i].dev_path = cam_devs[i];
        cpu_threads.emplace_back(worker_cpu_func, std::ref(cpu_ctxs[i]), std::ref(cpu_queue));
    }

    // ---- start stitching thread ----
    wctx.model_w = model_w;
    wctx.model_h = model_h;
    std::thread stitch_thread(cpu_warp_func, std::ref(wctx),
                              std::ref(cpu_queue), std::ref(npu_queue));

    // ---- start NPU inference workers ----
    std::vector<NPUWorkerCtx> npu_ctxs(NUM_NPU);
    std::vector<std::thread> npu_threads;
    for (int i = 0; i < NUM_NPU; i++) {
        npu_ctxs[i].id = i;
        npu_ctxs[i].model_path = model_path;
        npu_ctxs[i].core_mask = static_cast<rknn_core_mask>(1 << (i % 3));
    }
    for (int i = 0; i < NUM_NPU; i++) {
        npu_threads.emplace_back(worker_npu_func, std::ref(npu_ctxs[i]), std::ref(npu_queue));
    }

    printf("\n=== 4cam stitching + NPU inference running ===\n");
    printf("Press ESC or 'q' to exit\n\n");

    MjpegStreamer streamer(8080);
    streamer.start();

    // ---- main display loop ----
    auto t0 = std::chrono::steady_clock::now();
    int fcount = 0;
    int last_frame_id = -1;
    double fps_display = 0.0;

    while (!g_exit.load()) {
        int ready_idx = npu_queue.claim_ready();
        if (ready_idx >= 0) {
            int fid = npu_queue.meta[ready_idx]->frame_id;
            if (fid > last_frame_id) {
                last_frame_id = fid;
                int ret = imcopy(npu_queue.rga_buf[ready_idx], npu_queue.display_rga);
                if (ret == IM_STATUS_SUCCESS) {
                    dma_sync_device_to_cpu(npu_queue.display_fd);
                    cv::cvtColor(npu_queue.display_cv, npu_queue.display_cv, cv::COLOR_RGB2BGR);
                    char fps_text[32];
                    snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", fps_display);
                    cv::putText(npu_queue.display_cv, fps_text, cv::Point(10, 30),
                                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
                    streamer.push_frame(npu_queue.display_cv);
                }

                fcount++;
                auto t1 = std::chrono::steady_clock::now();
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                if (ms >= 2000) {
                    fps_display = fcount * 1000.0 / ms;
                    printf("[display] FPS: %.1f\n", fps_display);
                    fcount = 0;
                    t0 = t1;
                }
            }
            npu_queue.mark_free(ready_idx);
        }
    }

    // ---- cleanup ----
    printf("\nShutting down...\n");
    g_exit.store(true);
    streamer.stop();

    for (auto& ctx : cpu_ctxs) ctx.running.store(false);
    wctx.running.store(false);

    for (auto& ctx : npu_ctxs) {
        ctx.running.store(false);
    }

    for (auto& t : cpu_threads) if (t.joinable()) t.join();
    if (stitch_thread.joinable()) stitch_thread.join();
    for (auto& t : npu_threads) if (t.joinable()) t.join();

    cpu_queue.free_bufs();
    npu_queue.destroy();

    printf("Done.\n");
    return 0;
}
