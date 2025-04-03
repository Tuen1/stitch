#ifndef CYLINDER_PROJECTION_H
#define CYLINDER_PROJECTION_H

#include <opencv2/opencv.hpp>
#include <vector>

void apply_cylinder_projection(
    const cv::Mat& img,
    cv::Mat& output, 
    int rows,
    int output_cols,
    const std::vector<int>& target_indices,
    const std::vector<cv::Point>& source_coords);

#endif // CYLINDER_PROJECTION_H