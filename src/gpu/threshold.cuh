#pragma once

int findOtsuThreshold(const unsigned char* d_input, int width, int height);

void runThresholding(const unsigned char* h_input, unsigned char* h_output, int width, int height);

