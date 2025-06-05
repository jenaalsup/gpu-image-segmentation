#include <iostream>
#include <opencv2/opencv.hpp>
#include "gaussian.cuh"
#include "threshold.cuh"
#include "labeling.cuh"

int main() {
    std::cout << "GPU Segmentation demo starting..." << std::endl;

    // load image
    std::string input_path = "../data/embryo512.png";
    cv::Mat input = cv::imread(input_path, cv::IMREAD_GRAYSCALE);

    // gaussian blur
    cv::Mat blurred(input.size(), CV_8UC1);
    runGaussianBlur(input.data, blurred.data, input.cols, input.rows);
    std::string blurred_path = "../outputs/gpu/gaussian_output.png";
    cv::imwrite(blurred_path, blurred);
    std::cout << "Saved blurred image to " << blurred_path << std::endl;

    // thresholding
    cv::Mat binary(input.size(), CV_8UC1);
    runThresholding(blurred.data, binary.data, input.cols, input.rows);
    std::string binary_path = "../outputs/gpu/threshold_output.png";
    cv::imwrite(binary_path, binary);
    std::cout << "Saved thresholded image to " << binary_path << std::endl;


    // Labeling
    cv::Mat labels(binary.size(), CV_32S);
    runLabeling(binary.data, reinterpret_cast<int*>(labels.data), binary.cols, binary.rows);

    // remap the labels to be contiguous
    std::map<int, int> label_map;
    int next_id = 1;
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            int val = labels.at<int>(y, x);
            if (val > 0 && label_map.count(val) == 0) {
                label_map[val] = next_id++;
            }
        }
    }
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            int val = labels.at<int>(y, x);
            if (val > 0) {
                labels.at<int>(y, x) = label_map[val];
            }
        }
    }
    int num_labels = next_id - 1;

    // Post-processing: color and label
    cv::Mat output(binary.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            if (labels.at<int>(y, x) > 0) {
                output.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 230, 100);  // yellowish
            }
        }
    }

    for (int i = 1; i <= num_labels; i++) {
        cv::Mat mask = (labels == i);
        cv::Moments m = cv::moments(mask, true);
        if (m.m00 != 0) {
            int cx = static_cast<int>(m.m10 / m.m00);
            int cy = static_cast<int>(m.m01 / m.m00);
            cv::putText(output, std::to_string(i), {cx, cy},
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(191, 64, 191), 1);  // purple text
        }
    }

    std::string output_path = "../outputs/gpu/segmented_embryo512.png";
    cv::imwrite(output_path, output);
    std::cout << "Saved segmented image to " << output_path << std::endl;

    return 0;
}
