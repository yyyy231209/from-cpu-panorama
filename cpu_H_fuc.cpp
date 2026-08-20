#include "cpu_H_fuc.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>

static void compute_remap(const cv::Mat& M, int cw, int ch, cv::Mat& mx, cv::Mat& my) {
    cv::Mat iM = M.inv();
    mx.create(ch, cw, CV_32FC1);
    my.create(ch, cw, CV_32FC1);
    double a = iM.at<double>(0,0), b = iM.at<double>(0,1), c = iM.at<double>(0,2);
    double d = iM.at<double>(1,0), e = iM.at<double>(1,1), f = iM.at<double>(1,2);
    double g = iM.at<double>(2,0), h = iM.at<double>(2,1), ii = iM.at<double>(2,2);
    for (int y = 0; y < ch; y++) {
        float* px = mx.ptr<float>(y);
        float* py = my.ptr<float>(y);
        for (int x = 0; x < cw; x++) {
            double w = g * x + h * y + ii;
            px[x] = (float)((a * x + b * y + c) / w);
            py[x] = (float)((d * x + e * y + f) / w);
        }
    }
}

bool load_H_matrices(const std::string& dir,
                     cv::Mat& H12, cv::Mat& H32, cv::Mat& H42,
                     int& canvas_w, int& canvas_h,
                     cv::Mat& T, StitchMaps& maps)
{
    auto load_mat = [](const std::string& path) -> cv::Mat {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) { printf("ERROR: cannot open %s\n", path.c_str()); return cv::Mat(); }
        cv::Mat m(3, 3, CV_64F);
        size_t n = fread(m.data, sizeof(double), 9, f);
        fclose(f);
        if (n != 9) { printf("ERROR: short read %s\n", path.c_str()); return cv::Mat(); }
        return m;
    };
    H12 = load_mat(dir + "/H12.bin");
    H32 = load_mat(dir + "/H32.bin");
    H42 = load_mat(dir + "/H42.bin");
    if (H12.empty() || H32.empty() || H42.empty()) return false;

    int w = 320, h = 240;
    float corners[4][2] = {{0,0}, {0,(float)(h-1)}, {(float)(w-1),(float)(h-1)}, {(float)(w-1),0}};
    float xmin = INFINITY, ymin = INFINITY, xmax = -INFINITY, ymax = -INFINITY;
    auto add = [&](const cv::Mat& H) {
        for (int i = 0; i < 4; i++) {
            double a = H.at<double>(0,0), b = H.at<double>(0,1), c = H.at<double>(0,2);
            double d = H.at<double>(1,0), e = H.at<double>(1,1), f = H.at<double>(1,2);
            double g = H.at<double>(2,0), hh = H.at<double>(2,1), ii = H.at<double>(2,2);
            double ww = g * corners[i][0] + hh * corners[i][1] + ii;
            float ox = (float)((a * corners[i][0] + b * corners[i][1] + c) / ww);
            float oy = (float)((d * corners[i][0] + e * corners[i][1] + f) / ww);
            if (ox < xmin) xmin = ox;
            if (oy < ymin) ymin = oy;
            if (ox > xmax) xmax = ox;
            if (oy > ymax) ymax = oy;
        }
    };
    add(H12); add(cv::Mat::eye(3,3,CV_64F)); add(H32); add(H42);
    canvas_w = (int)(xmax - xmin + 1.5f);
    canvas_h = (int)(ymax - ymin + 1.5f);
    T = cv::Mat::eye(3, 3, CV_64F);
    T.at<double>(0, 2) = -xmin;
    T.at<double>(1, 2) = -ymin;

    printf("Canvas: %dx%d, computing remap maps...\n", canvas_w, canvas_h);
    cv::Mat Ms[4] = {T * H12, T, T * H32, T * H42};
    for (int i = 0; i < 4; i++)
        compute_remap(Ms[i], canvas_w, canvas_h, maps.map_x[i], maps.map_y[i]);

    maps.owner_map.create(canvas_h, canvas_w, CV_8UC1);
    maps.owner_map = 255;
    cv::Mat best_dist(canvas_h, canvas_w, CV_32FC1, cv::Scalar(1e30f));
    float cx = w / 2.0f, cy = h / 2.0f;

    for (int i = 0; i < 4; i++) {
        for (int y = 0; y < canvas_h; y++) {
            const float* px = maps.map_x[i].ptr<float>(y);
            const float* py = maps.map_y[i].ptr<float>(y);
            float* pdist = best_dist.ptr<float>(y);
            uint8_t* powner = maps.owner_map.ptr<uint8_t>(y);
            for (int x = 0; x < canvas_w; x++) {
                float sx = px[x], sy = py[x];
                if (sx >= 2 && sx < w-2 && sy >= 2 && sy < h-2) {
                    float dx = sx - cx, dy = sy - cy;
                    float dist = dx*dx + dy*dy;
                    if (dist < pdist[x]) {
                        pdist[x] = dist;
                        powner[x] = (uint8_t)i;
                    }
                }
            }
        }
    }

    maps.cw = canvas_w; maps.ch = canvas_h;
    printf("Remap ready.\n");
    return true;
}

void stitch_four_cpu(const cv::Mat& f1, const cv::Mat& f2,
                     const cv::Mat& f3, const cv::Mat& f4,
                     const StitchMaps& maps, cv::Mat& result)
{
    const cv::Mat* frames[4] = {&f1, &f2, &f3, &f4};
    int cw = maps.cw, ch = maps.ch;

    cv::Mat warped[4];
    for (int i = 0; i < 4; i++) {
        cv::remap(*frames[i], warped[i], maps.map_x[i], maps.map_y[i],
                  cv::INTER_LINEAR, cv::BORDER_REPLICATE, cv::Scalar(0,0,0));
    }

    result.create(ch, cw, CV_8UC3);
    result = cv::Scalar(0, 0, 0);

    for (int y = 0; y < ch; y++) {
        const uint8_t* owner = maps.owner_map.ptr<uint8_t>(y);
        uint8_t* out = result.ptr<uint8_t>(y);

        const uchar* srow[4];
        for (int i = 0; i < 4; i++) srow[i] = warped[i].ptr<uchar>(y);

        for (int x = 0; x < cw; x++) {
            int cam = owner[x];
            if (cam > 3) continue;
            const uchar* s = srow[cam];
            out[x*3]   = s[x*3];
            out[x*3+1] = s[x*3+1];
            out[x*3+2] = s[x*3+2];
        }
    }
}
