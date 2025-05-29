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
    std::vector<std::string> names = {"512", "1024", "2048"};
    for (const auto& name : names) {
        std::string input_path = "../data/embryo" + name + ".png";
        auto img = load_image(input_path);

        cv::Mat blurred = gaussian_blur(img, 15, 3.0);
        save_image("../outputs/blurred_embryo" + name + ".png", blurred);

        cv::Mat binary = threshold(blurred);
        save_image("../outputs/binary_embryo" + name + ".png", binary);

        cv::Mat labels = label_components(binary);
        double min, max;
        cv::minMaxLoc(labels, &min, &max);
        int num_labels = static_cast<int>(max);

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
        save_image("../outputs/segmented_embryo" + name + ".png", output);
    }

    return 0;
}
