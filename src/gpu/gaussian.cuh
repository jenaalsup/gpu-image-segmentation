#pragma once

void cudaCallGaussianBlur(const unsigned char* d_input, unsigned char* d_output, const float* d_kernel, int width, int height, int kernel_size);

void runGaussianBlur(const unsigned char* h_input, unsigned char* h_output, int width, int height, int kernel_size = 15, float sigma = 3.0f);
