#include "postprocess_crack.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <vector>

static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static inline float iou(const object_detect_result& a, const object_detect_result& b) {
    float x1 = std::max((float)a.box.left, (float)b.box.left);
    float y1 = std::max((float)a.box.top, (float)b.box.top);
    float x2 = std::min((float)a.box.right, (float)b.box.right);
    float y2 = std::min((float)a.box.bottom, (float)b.box.bottom);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = (a.box.right - a.box.left) * (a.box.bottom - a.box.top);
    float area_b = (b.box.right - b.box.left) * (b.box.bottom - b.box.top);
    return inter / (area_a + area_b - inter + 1e-6f);
}

static void nms(std::vector<object_detect_result>& dets, float threshold) {
    std::sort(dets.begin(), dets.end(), [](const object_detect_result& a, const object_detect_result& b) {
        return a.prop > b.prop;
    });
    for (size_t i = 0; i < dets.size(); i++) {
        if (dets[i].prop < 0) continue;
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (dets[j].prop < 0) continue;
            if (iou(dets[i], dets[j]) > threshold) {
                dets[j].prop = -1;
            }
        }
    }
    dets.erase(std::remove_if(dets.begin(), dets.end(), [](const object_detect_result& d) {
        return d.prop < 0;
    }), dets.end());
}

static void dfl_softmax(float* x, int length) {
    float max_val = x[0];
    for (int i = 1; i < length; i++) max_val = std::max(max_val, x[i]);
    float sum = 0;
    for (int i = 0; i < length; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < length; i++) x[i] /= sum;
}

int post_process_crack(rknn_output* outputs, rknn_app_context_t* app_ctx,
                       int model_w, int model_h,
                       float conf_threshold, float nms_threshold,
                       object_detect_result_list* od_results)
{
    od_results->count = 0;

    int strides[] = {8, 16, 32};
    const int DFL_LEN = 16;
    const int NUM_CLASSES = 1;

    float max_score[3] = {0, 0, 0};

    for (int si = 0; si < 3; si++) {
        int stride = strides[si];
        int grid_h = model_h / stride;
        int grid_w = model_w / stride;

        int off_bbox  = si * 3 + 0;
        int off_conf  = si * 3 + 1;
        int off_class = si * 3 + 2;

        if (off_class >= app_ctx->io_num.n_output) break;

        rknn_output& out_bbox  = outputs[off_bbox];
        rknn_output& out_conf  = outputs[off_conf];
        rknn_output& out_class = outputs[off_class];

        int8_t* bbox_data  = (int8_t*)out_bbox.buf;
        int8_t* conf_data  = (int8_t*)out_conf.buf;
        int8_t* class_data = (int8_t*)out_class.buf;

        float bbox_scale  = app_ctx->output_attrs[off_bbox].scale;
        int    bbox_zp    = app_ctx->output_attrs[off_bbox].zp;
        float conf_scale  = app_ctx->output_attrs[off_conf].scale;
        int    conf_zp    = app_ctx->output_attrs[off_conf].zp;
        float class_scale = app_ctx->output_attrs[off_class].scale;
        int    class_zp   = app_ctx->output_attrs[off_class].zp;

        for (int gy = 0; gy < grid_h; gy++) {
            for (int gx = 0; gx < grid_w; gx++) {
                int conf_idx = gy * grid_w + gx;
                float conf_val = ((float)conf_data[conf_idx] - conf_zp) * conf_scale;
                conf_val = sigmoid(conf_val);

                float class_val = ((float)class_data[conf_idx] - class_zp) * class_scale;
                class_val = sigmoid(class_val);

                float score = conf_val * class_val;
                if (score > max_score[si]) max_score[si] = score;
                if (score < conf_threshold) continue;

                float dfl_input[4 * DFL_LEN];
                for (int c = 0; c < 4; c++) {
                    for (int d = 0; d < DFL_LEN; d++) {
                        int idx = ((c * DFL_LEN + d) * grid_h + gy) * grid_w + gx;
                        dfl_input[c * DFL_LEN + d] = ((float)bbox_data[idx] - bbox_zp) * bbox_scale;
                    }
                }

                float box_dist[4] = {0, 0, 0, 0};
                for (int c = 0; c < 4; c++) {
                    float* group = &dfl_input[c * DFL_LEN];
                    dfl_softmax(group, DFL_LEN);
                    for (int d = 0; d < DFL_LEN; d++) {
                        box_dist[c] += group[d] * (float)d;
                    }
                }

                float x1 = ((float)gx - box_dist[0]) * stride;
                float y1 = ((float)gy - box_dist[1]) * stride;
                float x2 = ((float)gx + box_dist[2]) * stride;
                float y2 = ((float)gy + box_dist[3]) * stride;

                object_detect_result det;
                det.box.left   = (int)x1;
                det.box.top    = (int)y1;
                det.box.right  = (int)x2;
                det.box.bottom = (int)y2;
                det.prop = score;
                det.cls_id = 0;
                memset(det.keypoints, 0, sizeof(det.keypoints));

                std::vector<object_detect_result> single;
                single.push_back(det);
                nms(single, nms_threshold);

                for (auto& d : single) {
                    if (od_results->count < OBJ_NUMB_MAX_SIZE) {
                        od_results->results[od_results->count++] = d;
                    }
                }
            }
        }
    }

    printf("[crack] max_score: s8=%.4f s16=%.4f s32=%.4f thr=%.2f det=%d\n",
           max_score[0], max_score[1], max_score[2], conf_threshold, od_results->count);

    std::vector<object_detect_result> all_dets;
    for (int i = 0; i < od_results->count; i++) {
        all_dets.push_back(od_results->results[i]);
    }
    nms(all_dets, nms_threshold);
    od_results->count = 0;
    for (auto& d : all_dets) {
        if (od_results->count < OBJ_NUMB_MAX_SIZE) {
            od_results->results[od_results->count++] = d;
        }
    }

    return 0;
}
