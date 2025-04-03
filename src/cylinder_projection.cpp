#include "cylinder_projection.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <thrust/device_vector.h>

void apply_cylinder_projection(
    const cv::Mat& img,
    cv::Mat& output, 
    int rows,
    int output_cols,
    const std::vector<int>& target_indices,
    const std::vector<cv::Point>& source_coords) {

    // cv::Mat output(rows, output_cols, CV_8UC3, cv::Scalar(0, 0, 0));
    output.create(rows, output_cols, CV_8UC3);
    output.setTo(cv::Scalar(0, 0, 0));

    for (size_t i = 0; i < target_indices.size(); ++i) {
        int k = target_indices[i];
        int row = k / output_cols;
        int col = k % output_cols;

        if (row < 0 || row >= rows || col < 0 || col >= output_cols)
            continue;

        const cv::Point& pt = source_coords[i];
        int src_x = pt.x;
        int src_y = pt.y;

        if (src_x < 0 || src_x >= img.cols || src_y < 0 || src_y >= img.rows)
            continue;

        output.at<cv::Vec3b>(row, col) = img.at<cv::Vec3b>(src_y, src_x);
    }

}
