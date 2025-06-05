#include "gaussian.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// Maximum supported kernel size
#define MAX_KERNEL_SIZE 31

__constant__ float d_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

__global__ void gaussian_blur_kernel(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size, int half) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < half || x >= width - half || y < half || y >= height - half) return;

    float sum = 0.0f;
    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            int ix = x + dx;
            int iy = y + dy;
            float pixel = static_cast<float>(input[iy * width + ix]);
            float weight = d_kernel[(dy + half) * kernel_size + (dx + half)];
            sum += pixel * weight;
        }
    }
    output[y * width + x] = static_cast<unsigned char>(sum);
}

void launch_gaussian_blur(unsigned char* d_input, unsigned char* d_output, int width, int height, int kernel_size, float sigma) {
    if (kernel_size > MAX_KERNEL_SIZE) {
        std::cerr << "Kernel size too large!" << std::endl;
        return;
    }

    // Generate host-side Gaussian kernel
    int half = kernel_size / 2;
    float h_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
    float sum = 0.0f;

    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float value = expf(-(x * x + y * y) / (2.0f * sigma * sigma));
            h_kernel[(y + half) * kernel_size + (x + half)] = value;
            sum += value;
        }
    }

    // Normalize kernel
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        h_kernel[i] /= sum;
    }

    // Copy kernel to constant memory
    cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(float) * kernel_size * kernel_size);

    // Configure thread/block layout
    dim3 blockDim(16, 16);
    dim3 gridDim((width + 15) / 16, (height + 15) / 16);

    // Launch kernel
    gaussian_blur_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, kernel_size, half);
    cudaDeviceSynchronize();
}
