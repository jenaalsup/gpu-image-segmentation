#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "gaussian.cuh"
#include "threshold.cuh"
#include "labeling.cuh"

int main() {
    std::vector<std::string> images = {"512", "1024", "2048"};
    for (const auto& img_num : images) {
        auto start = std::chrono::high_resolution_clock::now();  // start timer

        std::string input_path = "../data/embryo" + img_num + ".png";
        cv::Mat input = cv::imread(input_path, cv::IMREAD_GRAYSCALE);

        const int width = input.cols;
        const int height = input.rows;
        const std::string out_dir = "../outputs/gpu/";

        // gaussian blur
        cv::Mat blurred(input.size(), CV_8UC1);
        runGaussianBlur(input.data, blurred.data, width, height);
        cv::imwrite(out_dir + "blurred_embryo" + img_num + ".png", blurred);

        // thresholding
        cv::Mat binary(input.size(), CV_8UC1);
        runThresholding(blurred.data, binary.data, width, height);
        cv::imwrite(out_dir + "binary_embryo" + img_num + ".png", binary);

        // labeling
        cv::Mat labels(binary.size(), CV_32S);
        runLabeling(binary.data, reinterpret_cast<int*>(labels.data), width, height);

        // remap the labels to be contiguous
        std::map<int, int> label_map;
        int next_id = 1;
        for (int i = 0; i < labels.rows * labels.cols; ++i) {
            int& val = labels.at<int>(i / width, i % width);
            if (val > 0) {
                if (!label_map.count(val)) label_map[val] = next_id++;
                val = label_map[val];
            }
        }
        int num_labels = next_id - 1;

        // post-processing: color and label overlay
        cv::Mat output(binary.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        for (int y = 0; y < labels.rows; y++) {
            for (int x = 0; x < labels.cols; x++) {
                if (labels.at<int>(y, x) > 0) {
                    output.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 230, 100); 
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
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(191, 64, 191), 1);  // purple labels
            }
        }
        std::string out_path = out_dir + "segmented_embryo" + img_num + ".png";
        cv::imwrite(out_path, output);
        std::cout << "saved segmented image to " << out_path << std::endl;

        auto end = std::chrono::high_resolution_clock::now();  // end timer
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "GPU segmentation time for embryo" << img_num << ": " << duration.count() << " ms" << std::endl;
    }
    return 0;
}
