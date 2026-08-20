#include "worker_npu.h"
#include "npu_queue.h"
#include "dma_alloc.h"
#include "rga/im2d.h"
#include <cstdio>
#include <cstring>
#include <thread>
#include <chrono>

#define CLS_COLORS { 0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFF8000, 0xFFFFFF }

static void rga_draw_rect(rga_buffer_t dst, int x, int y, int w, int h, int color, int thickness) {
    int t = thickness;
    im_rect rects[4] = {
        {x, y, w, t},
        {x, y + h - t, w, t},
        {x, y, t, h},
        {x + w - t, y, t, h},
    };
    for (int i = 0; i < 4; i++) {
        imfill(dst, rects[i], color);
    }
}

static int g_init_count = 0;

void worker_npu_func(NPUWorkerCtx& ctx, NPUQueue& queue) {
    if (g_init_count++ == 0) {
        init_post_process();
    }

    int ret = init_yolov8_model(ctx.model_path.c_str(), &ctx.rknn_ctx);
    if (ret != 0) {
        printf("[NPU%d] init model fail ret=%d\n", ctx.id, ret);
        g_init_count--;
        return;
    }
    ctx.model_w = ctx.rknn_ctx.model_width;
    ctx.model_h = ctx.rknn_ctx.model_height;
    printf("[NPU%d] ready, input=%dx%d core=%d\n",
           ctx.id, ctx.model_w, ctx.model_h, (int)ctx.core_mask);

    object_detect_result_list od_results;
    image_buffer_t src_image;

    unsigned int colors[] = CLS_COLORS;
    int ncolors = sizeof(colors) / sizeof(colors[0]);

    while (ctx.running.load()) {
        int buf_idx = queue.claim_filled();
        if (buf_idx < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        memset(&src_image, 0, sizeof(src_image));
        src_image.width = ctx.model_w;
        src_image.height = ctx.model_h;
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.virt_addr = (unsigned char*)queue.virt[buf_idx];
        src_image.width_stride = ctx.model_w * 3;
        src_image.height_stride = ctx.model_h;
        src_image.size = src_image.width_stride * src_image.height;
        src_image.fd = queue.fd[buf_idx];

        ret = inference_yolov8_model(&ctx.rknn_ctx, &src_image, &od_results);
        if (ret != 0) {
            printf("[NPU%d] inference fail ret=%d\n", ctx.id, ret);
            queue.mark_free(buf_idx);
            continue;
        }

        rga_buffer_t rga_buf = queue.rga_buf[buf_idx];

        if (od_results.count > 0) {
            printf("[NPU%d] %d detections:", ctx.id, od_results.count);
            for (int i = 0; i < od_results.count; i++) {
                printf(" %.1f%%", od_results.results[i].prop * 100);
            }
            printf("\n");
        }

        for (int i = 0; i < od_results.count; i++) {
            object_detect_result* det = &od_results.results[i];
            int x1 = det->box.left;
            int y1 = det->box.top;
            int x2 = det->box.right;
            int y2 = det->box.bottom;

            int color = colors[det->cls_id % ncolors];
            rga_draw_rect(rga_buf, x1, y1, x2 - x1, y2 - y1, color, 2);
        }

        queue.mark_ready(buf_idx);
    }

    release_yolov8_model(&ctx.rknn_ctx);
    if (--g_init_count == 0) {
        deinit_post_process();
    }
    printf("[NPU%d] stopped\n", ctx.id);
}
