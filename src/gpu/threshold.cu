#include <cuda_runtime.h>
#include <numeric>
#include <vector>
#include <algorithm>
#include "threshold.cuh"

static constexpr int THREADS_PER_BLOCK = 256;

__global__
void histogramKernel(const unsigned char* input, int* hist, int width, int height, int ignore_below) {
    // each thread is responsible for one pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    if (idx >= total_pixels) return;
    unsigned char val = input[idx];
    // if the pixel value is greater than the ignore_below threshold, increment the histogram at that index
    if (val >= ignore_below) atomicAdd(&hist[val], 1);
}

__global__
void binarizeKernel(const unsigned char* input, unsigned char* output, int width, int height, int threshold) {
    // each thread is responsible for one pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    if (idx >= total_pixels) return;
    // if the pixel value is greater than the threshold, set it to 255, otherwise set it to 0
    output[idx] = input[idx] > threshold ? 255 : 0;
}

// note: this function is almost identical to the CPU version, but it parallelizes computing the histogram 
int findOtsuThreshold(const unsigned char* d_input, int width, int height) {
    // ignore dark pixels below 15/256 when computing the histogram as they are part of the background
    const int ignore_below = 15;
    int* d_hist;
    cudaMalloc(&d_hist, 256 * sizeof(int)); // one bin for each possible pixel value (256 total)
    cudaMemset(d_hist, 0, 256 * sizeof(int));

    // compute the histogram of the intensity values in the image
    int total_pixels = width * height;
    int num_blocks = (total_pixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    histogramKernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_hist, width, height, ignore_below);
    cudaDeviceSynchronize();

    std::vector<int> h_hist(256);
    cudaMemcpy(h_hist.data(), d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_hist);

    int total = std::accumulate(h_hist.begin() + ignore_below, h_hist.end(), 0);
    if (total == 0) return 0;  // no valid pixels

    // total weighted sum of all pixel intensities
    double total_sum = 0.0;
    for (int i = 0; i < 256; i++) total_sum += i * h_hist[i];

    double sum_bg = 0.0;
    int weight_bg = 0;
    double max_var = 0.0;
    int best_thresh = ignore_below;

    // iterate through all possible thresholds to determine the best one
    for (int t = ignore_below; t < 256; t++) {
        weight_bg += h_hist[t];
        if (weight_bg == 0) continue;
        int weight_fg = total - weight_bg;
        if (weight_fg == 0) break;

        sum_bg += t * h_hist[t];
        double mean_bg = sum_bg / weight_bg;
        double mean_fg = (total_sum - sum_bg) / weight_fg;

        double var = weight_bg * weight_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
        if (var > max_var) {
            max_var = var;
            best_thresh = t;
        }
    }
    return best_thresh;
}

void runThresholding(const unsigned char* h_input, unsigned char* h_output, int width, int height) {
    size_t img_size = width * height * sizeof(unsigned char);

    // allocate gpu memory and copy image from host to device
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);
    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);

    // calculate the threshold
    int threshold = findOtsuThreshold(d_input, width, height);

    // apply the threshold to the image to binarize it
    int total_pixels = width * height;
    int num_blocks = (total_pixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    binarizeKernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, width, height, threshold);
    cudaDeviceSynchronize();

    // copy the binarized image from device back to the host
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
