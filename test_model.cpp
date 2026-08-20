#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include "yolov8.h"
#include "common.h"
#include "image_utils.h"
#include "file_utils.h"
#include "postprocess.h"
#include "postprocess_crack.h"

int main(int argc, char** argv) {
    const char* img_path = argc > 1 ? argv[1] : "train_00093.jpg";
    const char* model_path = argc > 2 ? argv[2] : "best.rknn";

    printf("=== Model Test ===\n");
    printf("Image: %s\n", img_path);
    printf("Model: %s\n\n", model_path);

    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        printf("ERROR: cannot read %s\n", img_path);
        return 1;
    }
    printf("Image: %dx%d channels=%d\n", img.cols, img.rows, img.channels());

    init_post_process();

    rknn_app_context_t app_ctx;
    memset(&app_ctx, 0, sizeof(app_ctx));
    int ret = init_yolov8_model(model_path, &app_ctx);
    if (ret != 0) {
        printf("ERROR: init model fail ret=%d\n", ret);
        return 1;
    }
    printf("Model: %dx%d ch=%d is_quant=%d outputs=%d\n",
           app_ctx.model_width, app_ctx.model_height,
           app_ctx.model_channel, app_ctx.is_quant,
           app_ctx.io_num.n_output);

    // Resize to model input size
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(app_ctx.model_width, app_ctx.model_height));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    image_buffer_t src;
    memset(&src, 0, sizeof(src));
    src.width = app_ctx.model_width;
    src.height = app_ctx.model_height;
    src.width_stride = src.width * 3;
    src.height_stride = src.height;
    src.format = IMAGE_FORMAT_RGB888;
    src.virt_addr = resized.data;
    src.size = src.width_stride * src.height;

    object_detect_result_list results;
    ret = inference_yolov8_model(&app_ctx, &src, &results);
    if (ret != 0) {
        printf("ERROR: inference fail ret=%d\n", ret);
        release_yolov8_model(&app_ctx);
        return 1;
    }

    printf("\n=== Results: %d detections ===\n", results.count);
    for (int i = 0; i < results.count && i < 20; i++) {
        auto& d = results.results[i];
        printf("  [%d] cls=%d conf=%.4f box=[%d,%d,%d,%d] w=%d h=%d\n",
               i, d.cls_id, d.prop,
               d.box.left, d.box.top, d.box.right, d.box.bottom,
               d.box.right - d.box.left, d.box.bottom - d.box.top);
    }
    if (results.count > 20) printf("  ... +%d more\n", results.count - 20);

    // Draw top detections
    for (int i = 0; i < results.count && i < 20; i++) {
        auto& d = results.results[i];
        cv::rectangle(resized, cv::Point(d.box.left, d.box.top),
                      cv::Point(d.box.right, d.box.bottom), cv::Scalar(0, 255, 0), 2);
        char buf[64];
        snprintf(buf, sizeof(buf), "%.2f", d.prop);
        cv::putText(resized, buf, cv::Point(d.box.left, d.box.top - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
    }

    cv::Mat out;
    cv::cvtColor(resized, out, cv::COLOR_RGB2BGR);
    cv::imwrite("test_result.jpg", out);
    printf("\nSaved: test_result.jpg\n");

    release_yolov8_model(&app_ctx);
    deinit_post_process();
    return 0;
}
