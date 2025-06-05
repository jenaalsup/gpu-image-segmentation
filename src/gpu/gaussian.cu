#include <cuda_runtime.h>
#include <vector>
#include <numbers>
#include "gaussian.cuh"

static constexpr int32_t THREADS_PER_BLOCK = 256;

static float gaussian(float x, float mean, float std) {
    float pi = std::numbers::pi_v<float>;
    return (1.0f / (std * sqrtf(2 * pi))) * expf(-0.5f * powf((x - mean) / std, 2));
}

__global__
void gaussianBlurKernel(const unsigned char* input, unsigned char* output, const float* kernel, int width, int height, int kernel_size, int half) {
    // each thread is responsible for one pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    if (idx >= total_pixels) return;
    int x = idx % width; // column
    int y = idx / width; // row

    // skip border pixels where the gaussian kernel would run off the edge of the image
    if (x < half || x >= width - half || y < half || y >= height - half) return;

    // apply the gaussian kernel to each pixel in the local patch of cells around it
    float sum = 0.0f;
    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            float pixel = static_cast<float>(input[(y + dy) * width + (x + dx)]);
            float weight = kernel[(dy + half) * kernel_size + (dx + half)];
            sum += pixel * weight;
        }
    }

    // write the blurred pixel value to the output image
    output[y * width + x] = static_cast<unsigned char>(sum);
}

void cudaCallGaussianBlur(const unsigned char* d_input, unsigned char* d_output, const float* d_kernel, int width, int height, int kernel_size) {
    int half = kernel_size / 2;
    int total_pixels = width * height;
    int num_blocks = (total_pixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gaussianBlurKernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, d_kernel, width, height, kernel_size, half);
    cudaDeviceSynchronize();
}

void runGaussianBlur(const unsigned char* h_input, unsigned char* h_output, int width, int height, int kernel_size, float sigma) {
    size_t img_size = width * height * sizeof(unsigned char);

    // allocate gpu memory and copy image from host to device
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);
    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);

    // construct gaussian weight matrix to apply to each pixel for blurring
    int half = kernel_size / 2;
    std::vector<float> h_kernel(kernel_size * kernel_size);
    float sum = 0.0f;
    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            float val = gaussian(dx, 0, sigma) * gaussian(dy, 0, sigma);
            h_kernel[(dy + half) * kernel_size + (dx + half)] = val;
            sum += val;
        }
    }
    for (float& val : h_kernel) val /= sum;
    float* d_kernel;
    cudaMalloc(&d_kernel, h_kernel.size() * sizeof(float));
    cudaMemcpy(d_kernel, h_kernel.data(), h_kernel.size() * sizeof(float), cudaMemcpyHostToDevice);

    // call the gpu kernel to blur the image
    cudaCallGaussianBlur(d_input, d_output, d_kernel, width, height, kernel_size);

    // copy the blurred image from device back to the host
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}
