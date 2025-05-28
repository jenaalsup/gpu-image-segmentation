#pragma once
#include <string>
#include <opencv2/core.hpp>

cv::Mat load_image(const std::string& path);
void save_image(const std::string& path, const cv::Mat& image);
