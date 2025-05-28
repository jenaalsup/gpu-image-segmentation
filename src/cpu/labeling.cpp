#include "labeling.hpp"
#include <stack>

bool is_foreground(const cv::Mat& img, int y, int x) {
    return img.at<uchar>(y, x) == 255;
}

cv::Mat label_components(const cv::Mat& binary) {
    cv::Mat labels = cv::Mat::zeros(binary.size(), CV_32S);
    int label = 1;

    int rows = binary.rows;
    int cols = binary.cols;

    // 8-connected directions
    int dx[] = {-1, 0, 1, -1, 0, 1, -1, 1};
    int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (is_foreground(binary, y, x) && labels.at<int>(y, x) == 0) {
                // start a new region
                std::stack<std::pair<int, int>> s;
                s.push({y, x});
                labels.at<int>(y, x) = label;

                while (!s.empty()) {
                    auto [cy, cx] = s.top();
                    s.pop();

                    // check all 8 neighbors
                    for (int k = 0; k < 8; k++) {
                        int ny = cy + dy[k];
                        int nx = cx + dx[k];

                        if (ny >= 0 && ny < rows && nx >= 0 && nx < cols &&
                            is_foreground(binary, ny, nx) &&
                            labels.at<int>(ny, nx) == 0) {
                            labels.at<int>(ny, nx) = label;
                            s.push({ny, nx});
                        }
                    }
                }
                label++;
            }
        }
    }

    return labels;
}
