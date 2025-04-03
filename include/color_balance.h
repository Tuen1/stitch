#ifndef COLOR_BALANCE_H
#define COLOR_BALANCE_H

#include <opencv2/opencv.hpp>

class ColorBlance {
public:
    static cv::Mat colorblance_left(const cv::Mat& img_L, const cv::Mat& img_M, int W);
    static cv::Mat colorblance_right(const cv::Mat& img_M, const cv::Mat& img_R, int W);
    static void color_balance_peizhun_left(const cv::Mat& img_L, const cv::Mat& img_M, int W, int y_s_1, int y_x_1,
                                         const cv::Mat& H_left, cv::Mat& left_warped, cv::Mat& canvas);
    static void color_balance_peizhun_right(const cv::Mat& img_M, const cv::Mat& img_R, int W, int y_s_1, int y_x_1,
                                          const cv::Mat& H_right, cv::Mat& canvas, cv::Mat& right_warped);
};

#endif // COLOR_BALANCE_H