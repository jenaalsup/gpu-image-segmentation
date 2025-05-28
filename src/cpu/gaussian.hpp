#pragma once
#include <opencv2/core.hpp>

cv::Mat gaussian_blur(const cv::Mat& input, int kernel_size, double sigma);
