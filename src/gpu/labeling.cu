#include <cuda_runtime.h>
#include <iostream>
#include "labeling.cuh"

static constexpr int THREADS_PER_BLOCK = 256;

__global__
void initializeLabels(const unsigned char* input, int* labels, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    if (idx >= total_pixels) return;

    labels[idx] = (input[idx] == 255) ? idx + 1 : 0; // label foreground with unique ID
}

__global__
void propagateLabels(const unsigned char* input, int* labels, int width, int height, bool* changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    if (idx >= total_pixels) return;

    int x = idx % width;
    int y = idx / width;

    if (input[idx] != 255) return;

    int label = labels[idx];
    int min_label = label;

    // 8-neighbor directions
    int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};

    for (int k = 0; k < 8; k++) {
        int nx = x + dx[k];
        int ny = y + dy[k];
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            int n_idx = ny * width + nx;
            if (input[n_idx] == 255 && labels[n_idx] < min_label && labels[n_idx] > 0) {
                min_label = labels[n_idx];
            }
        }
    }

    if (min_label < label) {
        labels[idx] = min_label;
        *changed = true;
    }
}

__global__
void compressLabels(int* labels, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    if (idx >= total_pixels) return;

    if (labels[idx] > 0) {
        int root = labels[idx];
        while (labels[root - 1] != root) {
            root = labels[root - 1];
        }
        labels[idx] = root;
    }
}

void runLabeling(const unsigned char* h_input, int* h_labels, int width, int height) {
    size_t img_size = width * height * sizeof(unsigned char);
    size_t label_size = width * height * sizeof(int);

    unsigned char* d_input;
    int* d_labels;
    bool* d_changed;
    bool h_changed;

    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_labels, label_size);
    cudaMalloc(&d_changed, sizeof(bool));
    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);

    int total_pixels = width * height;
    int num_blocks = (total_pixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Step 1: Initialize labels
    initializeLabels<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_labels, width, height);
    cudaDeviceSynchronize();

    // Step 2: Iterative label propagation
    int max_iters = 100;
    int iter = 0;

    do {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        propagateLabels<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_labels, width, height, d_changed);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        iter++;

        if (iter > max_iters) {
            std::cerr << "runLabeling exceeded max iterations!" << std::endl;
            break;
        }
    } while (h_changed);

    // Step 3: Compress labels to canonical root
    compressLabels<<<num_blocks, THREADS_PER_BLOCK>>>(d_labels, width, height);
    cudaDeviceSynchronize();

    // Step 4: Copy labels back to host
    cudaMemcpy(h_labels, d_labels, label_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_labels);
    cudaFree(d_changed);
}
