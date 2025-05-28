#include <iostream>
#include "image_io.hpp"

int main() {
    std::string input_path = "../data/embryo512.png";
    auto img = load_image(input_path);
    std::cout << "Loaded image of size: " << img.cols << " x " << img.rows << std::endl;

    save_image("../outputs/copy_embryo512.png", img);
    return 0;
}
