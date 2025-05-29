#include "image_io.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>

cv::Mat load_image(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) exit(1);
    return image;
}

void save_image(const std::string& path, const cv::Mat& image) {
    if (!cv::imwrite(path, image)) exit(1);
}
