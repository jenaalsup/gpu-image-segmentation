#include <iostream>
#include "image_io.hpp"
#include "gaussian.hpp"
#include "threshold.hpp"
#include "labeling.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main() {
    std::string input_path = "../data/embryo2048.png";
    auto img = load_image(input_path);

    // part 1: gaussian blur (kernel)
    cv::Mat blurred = gaussian_blur(img, 15, 5.0);
    save_image("../outputs/blurred_embryo2048.png", blurred);

    // part 2: threshold (kernel)
    cv::Mat binary = threshold(blurred);
    save_image("../outputs/binary_embryo2048.png", binary);

    // part 3: label connected components (kernel)
    cv::Mat labels = label_components(binary);
    double min, max;
    cv::minMaxLoc(labels, &min, &max);
    int num_labels = static_cast<int>(max);

    // part 4: display segmented image with label numbers (not a kernel)
    cv::Mat output(binary.size(), CV_8UC3, cv::Scalar(0, 0, 0));  // black background
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            if (labels.at<int>(y, x) > 0) {
                output.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 230, 100);  // cyan
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
    save_image("../outputs/segmented_embryo2048.png", output);

    return 0;
}
