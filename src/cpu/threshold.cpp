#include "threshold.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>

cv::Mat threshold(const cv::Mat& input) {
    // compute histogram of pixel intensities (256 bins)
    int hist[256] = {0};
    int total_pixels = input.rows * input.cols;

    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
            uchar value = input.at<uchar>(y, x);
            hist[value]++;
        }
    }

    // compute otsu's threshold
    double sum_all = 0.0;
    for (int i = 0; i < 256; i++)
        sum_all += i * hist[i];

    double sum_bg = 0.0;
    int weight_bg = 0;
    int weight_fg = 0;
    double max_between_var = 0.0;
    int best_threshold = 0;

    // try every possible threshold
    for (int t = 0; t < 256; t++) {
        weight_bg += hist[t];
        if (weight_bg == 0) continue;

        weight_fg = total_pixels - weight_bg;
        if (weight_fg == 0) break;

        sum_bg += t * hist[t];
        double mean_bg = sum_bg / weight_bg;
        double mean_fg = (sum_all - sum_bg) / weight_fg;

        // maximize the between-class variance (difference between the means of the two bright / dark groups)
        double between_var = weight_bg * weight_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
        if (between_var > max_between_var) {
            max_between_var = between_var;
            best_threshold = t;
        }
    }

    std::cout << "Threshold: " << best_threshold << std::endl;

    // apply the threshold to create binary image
    cv::Mat binary = input.clone();
    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
            uchar pixel = input.at<uchar>(y, x);
            binary.at<uchar>(y, x) = (pixel >= best_threshold) ? 255 : 0;
        }
    }

    return binary;
}
