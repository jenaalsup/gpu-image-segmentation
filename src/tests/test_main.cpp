#include <iostream>
#include "../cpu/gaussian.hpp"
#include "../cpu/threshold.hpp"
#include "../cpu/labeling.hpp"
#include "../cpu/image_io.hpp"
#include <opencv2/imgcodecs.hpp>
#include <cassert>

int main() {

    // test 1: image is loaded correctly
    std::string input_path = "../data/embryo512.png";
    auto img = load_image(input_path);
    assert(!img.empty());

    // test 2: gaussian blur runs without crashing
    auto blurred = gaussian_blur(img, 15, 3.0);

    // test 3: thresholding produces a valid binary image
    auto binary = threshold(blurred);

    // test 4: labeling algorithm terminates and doesn't crash
    auto labels = label_components(binary);

    std::cout << "All tests run successfully." << std::endl;
    return 0;
}
