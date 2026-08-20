#pragma once
#include "yolov8.h"
#include "postprocess.h"

int post_process_crack(rknn_output* outputs, rknn_app_context_t* app_ctx,
                       int model_w, int model_h,
                       float conf_threshold, float nms_threshold,
                       object_detect_result_list* od_results);
