#include <iostream>
#include <vector>
#include "image_io.hpp"
#include "gaussian.hpp"
#include "threshold.hpp"
#include "labeling.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main() {
    std::vector<std::string> images = {"512", "1024", "2048"};
    for (const auto& img_num : images) {
        std::string input_path = "../data/embryo" + img_num + ".png";
        auto img = load_image(input_path);

        // kernel 1: gaussian blur
        cv::Mat blurred = gaussian_blur(img, 15, 3.0);
        save_image("../outputs/blurred_embryo" + img_num + ".png", blurred);

        // kernel 2: thresholding
        cv::Mat binary = threshold(blurred);
        save_image("../outputs/binary_embryo" + img_num + ".png", binary);

        // kernel 3: labeling components (nuclei)
        cv::Mat labels = label_components(binary);
        double min, max;
        cv::minMaxLoc(labels, &min, &max);
        int num_labels = static_cast<int>(max);

        // post-processing: color the nuclei and label them
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
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(191, 64, 191), 1);
            }
        }
        save_image("../outputs/segmented_embryo" + img_num + ".png", output);
        std::cout << "saved segmented image to outputs/segmented_embryo" + img_num + ".png" << std::endl;
    }

    return 0;
}
