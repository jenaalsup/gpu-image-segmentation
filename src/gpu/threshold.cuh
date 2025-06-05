#pragma once

#include <opencv2/core.hpp>

// Launches GPU thresholding pipeline and returns the binarized output.
// Expects input and output to be grayscale 8-bit images (CV_8UC1).
void runThresholding(const unsigned char* h_input, unsigned char* h_output, int width, int height);

// Computes histogram from an image on the GPU
int findOtsuThreshold(const unsigned char* d_input, int width, int height);

// Binarizes image on GPU using the provided threshold
void binarizeImage(const unsigned char* d_input, unsigned char* d_output, int width, int height, int threshold);
