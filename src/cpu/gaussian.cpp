#include "gaussian.hpp"
#include <cmath>

// create 2d gaussian kernel matrix
static cv::Mat create_gaussian_kernel(int ksize, double sigma) {
    int half = ksize / 2;
    cv::Mat kernel(ksize, ksize, CV_64F); // holds the weights of the kernel
    double sum = 0.0;

    // fill in each element of the kernel going about the center 
    // PARALLELIZE THIS: assign each pixel to a thread
    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            double gaussian_value = std::exp(-(dx * dx + dy * dy) / (2 * sigma * sigma)); // closer cells get higher weights
            kernel.at<double>(dy + half, dx + half) = gaussian_value;
            sum += gaussian_value;
        }
    }

    kernel /= sum; // normalize the kernel so that all weights add up to 1
    return kernel;
}

// apply gaussian blur to image using 2d convolution
cv::Mat gaussian_blur(const cv::Mat& input, int kernel_size, double sigma) {
    int half = kernel_size / 2;
    cv::Mat kernel = create_gaussian_kernel(kernel_size, sigma);
    cv::Mat output = input.clone();
    for (int y = half; y < input.rows - half; y++) {
        for (int x = half; x < input.cols - half; x++) {
            double blurred = 0.0;
            // blur the pixel by applying the kernel to the local patch of cells around it
            for (int dy = -half; dy <= half; dy++) {
                for (int dx = -half; dx <= half; dx++) {
                    uchar pixel = input.at<uchar>(y + dy, x + dx);
                    double weight = kernel.at<double>(dy + half, dx + half);
                    blurred += pixel * weight;
                }
            }
            output.at<uchar>(y, x) = static_cast<uchar>(blurred);
        }
    }
    return output;
}
