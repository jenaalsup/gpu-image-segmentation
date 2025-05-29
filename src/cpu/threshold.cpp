#include "threshold.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

cv::Mat threshold(const cv::Mat& input) {
    const int ignore_below = 15;

    // part 1: collect all pixel intensities into a flat vector
    std::vector<uchar> pixels;
    if (input.isContinuous()) {
        pixels.assign(input.datastart, input.dataend);
    } else {
        for (int y = 0; y < input.rows; y++) {
            pixels.insert(pixels.end(), input.ptr<uchar>(y), input.ptr<uchar>(y) + input.cols);
        }
    }
    std::sort(pixels.begin(), pixels.end());
    // clip extremes at 1% and 99%
    int n = pixels.size();
    uchar low_clip = pixels[n * 1 / 100];
    uchar high_clip = pixels[n * 99 / 100];

    // part 2: contrast stretch using OpenCV normalization
    cv::Mat clipped = input.clone();
    cv::threshold(clipped, clipped, high_clip, high_clip, cv::THRESH_TRUNC);
    cv::threshold(clipped, clipped, low_clip, low_clip, cv::THRESH_TOZERO);
    cv::normalize(clipped, clipped, 0, 255, cv::NORM_MINMAX);

    // part 3: build histogram ignoring dark values
    int hist[256] = {0}, total_pixels = 0;
    for (int y = 0; y < clipped.rows; y++) {
        for (int x = 0; x < clipped.cols; x++) {
            uchar v = clipped.at<uchar>(y, x);
            if (v >= ignore_below) {
                hist[v]++;
                total_pixels++;
            }
        }
    }
    // no valid pixels
    if (total_pixels == 0) return clipped.clone();

    // step 4: otsu's method for thresholding
    double sum_all = 0.0;
    for (int i = 0; i < 256; i++) sum_all += i * hist[i];

    double sum_bg = 0.0;
    int weight_bg = 0;
    double max_var = 0.0;
    int best_thresh = ignore_below;

    // test all thresholds from ignore_below to 255
    for (int t = ignore_below; t < 256; t++) {
        weight_bg += hist[t];
        if (weight_bg == 0) continue;

        int weight_fg = total_pixels - weight_bg;
        if (weight_fg == 0) break;

        sum_bg += t * hist[t];
        double mean_bg = sum_bg / weight_bg;
        double mean_fg = (sum_all - sum_bg) / weight_fg;

        double var = weight_bg * weight_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
        if (var > max_var) {
            max_var = var;
            best_thresh = t;
        }
    }

    // part 5: binarize
    cv::Mat binary;
    cv::threshold(clipped, binary, best_thresh, 255, cv::THRESH_BINARY);

    return binary;
}
