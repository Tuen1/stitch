#ifndef STITCH_H
#define STITCH_H

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void stitch_cuda(const cv::Mat& img1, const cv::Mat& img2, const float* weights, cv::Mat& result);



#endif // STITCH_H