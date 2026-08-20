#pragma once
#include <opencv2/opencv.hpp>

struct StitchMaps {
    cv::Mat map_x[4];
    cv::Mat map_y[4];
    cv::Mat owner_map;
    int cw = 0, ch = 0;
};

bool load_H_matrices(const std::string& dir,
                     cv::Mat& H12, cv::Mat& H32, cv::Mat& H42,
                     int& canvas_w, int& canvas_h,
                     cv::Mat& T, StitchMaps& maps);

void stitch_four_cpu(const cv::Mat& f1, const cv::Mat& f2,
                     const cv::Mat& f3, const cv::Mat& f4,
                     const StitchMaps& maps, cv::Mat& result);
