#include <iostream>
#include <opencv2/opencv.hpp>
#include "gaussian.cuh"

int main() {
    std::cout << "GPU Segmentation demo starting..." << std::endl;

    std::string input_path = "../data/embryo512.png";
    cv::Mat input = cv::imread(input_path, cv::IMREAD_GRAYSCALE);

    cv::Mat output(input.size(), CV_8UC1);
    runGaussianBlur(input.data, output.data, input.cols, input.rows);

    std::string output_path = "../outputs/gpu/gaussian_output.png";
    cv::imwrite(output_path, output);
    std::cout << "Saved blurred image to " << output_path << std::endl;

    return 0;
}
