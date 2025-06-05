#include <iostream>
#include <opencv2/opencv.hpp>
#include "gaussian.cuh"
#include "threshold.cuh"
#include "labeling.cuh"

int main() {
    std::cout << "GPU Segmentation demo starting..." << std::endl;

    // load image
    std::string input_path = "../data/embryo512.png";
    cv::Mat input = cv::imread(input_path, cv::IMREAD_GRAYSCALE);

    // gaussian blur
    cv::Mat blurred(input.size(), CV_8UC1);
    runGaussianBlur(input.data, blurred.data, input.cols, input.rows);
    std::string blur_path = "../outputs/gpu/gaussian_output.png";
    cv::imwrite(blur_path, blurred);
    std::cout << "Saved blurred image to " << blur_path << std::endl;

    // thresholding
    cv::Mat binary(input.size(), CV_8UC1);
    runThresholding(blurred.data, binary.data, input.cols, input.rows);
    std::string binary_path = "../outputs/gpu/threshold_output.png";
    cv::imwrite(binary_path, binary);
    std::cout << "Saved thresholded image to " << binary_path << std::endl;

    return 0;
}
