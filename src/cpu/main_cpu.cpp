#include <iostream>
#include "image_io.hpp"
#include "gaussian.hpp"
#include "threshold.hpp"

int main() {
    std::string input_path = "../data/embryo512.png";
    auto img = load_image(input_path);
    std::cout << "Loaded image of size: " << img.cols << " x " << img.rows << std::endl;

    // gaussian blur
    cv::Mat blurred = gaussian_blur(img, 11, 2.0);
    save_image("../outputs/blurred_embryo512.png", blurred);

    // threshold
    cv::Mat binary = threshold(blurred);
    save_image("../outputs/binary_embryo512.png", binary);
    
    return 0;
}
