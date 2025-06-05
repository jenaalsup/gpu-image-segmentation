#include <iostream>
#include <opencv2/opencv.hpp>
#include "gaussian.cuh"  // include your CUDA blur header

int main() {
    std::cout << "GPU Segmentation demo starting..." << std::endl;

    // 1. Load input image (grayscale)
    std::string input_path = "data/embryo512.png";
    cv::Mat input = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        std::cerr << "Failed to load input image: " << input_path << std::endl;
        return 1;
    }

    int width = input.cols;
    int height = input.rows;
    size_t img_size = width * height * sizeof(unsigned char);

    // 2. Allocate output container
    cv::Mat output(height, width, CV_8UC1);

    // 3. Allocate GPU memory
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);

    // 4. Copy input image to GPU
    cudaMemcpy(d_input, input.data, img_size, cudaMemcpyHostToDevice);

    // 5. Launch Gaussian blur
    int kernel_size = 15;
    float sigma = 3.0f;
    launch_gaussian_blur(d_input, d_output, width, height, kernel_size, sigma);

    // 6. Copy result back to CPU
    cudaMemcpy(output.data, d_output, img_size, cudaMemcpyDeviceToHost);

    // 7. Save output image
    std::string output_path = "outputs/gaussian_output.png";
    cv::imwrite(output_path, output);
    std::cout << "Saved blurred image to: " << output_path << std::endl;

    // 8. Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
