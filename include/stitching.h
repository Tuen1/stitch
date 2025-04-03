#ifndef STITCHING_H
#define STITCHING_H

#include <opencv2/opencv.hpp>
#include "readData.h"
#include <vector>
using namespace cv;
using namespace std;

void stitch(const cv::Mat& img1, const cv::Mat& img2, const float* weights, cv::Mat& result);

cv::Mat process_and_stitch_images(const cv::Mat& img_left, const cv::Mat& img_mid, const cv::Mat& img_right, 
    const CalibrationData& data, const vector<Point>& source_coords,const cv::Mat& H_left,const cv::Mat& H_right,
    vector<int>& target_indices, const int& rows, const int& output_cols,const int& y_s_1, const int& y_x_1,
    const int& W,const float* weights_left, float* weights_right);

#endif // STITCHING_H