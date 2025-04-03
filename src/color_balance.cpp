#include "color_balance.h"
#include <algorithm>
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>  // 包含 CUDA 图像变换相关函数（包括 warpPerspective）
#include <opencv2/core/cuda.hpp>    // 包含 CUDA 数据结构（如 GpuMat）
using namespace cv;
using namespace std;
using namespace std::chrono;
template<typename T>
const T& clamp(const T& value, const T& low, const T& high) {
    return (value < low) ? low : (value > high) ? high : value;
}

Mat ColorBlance::colorblance_left(const Mat& img_L, const Mat& img_M, int W) {

    // 计算亮度差
    Scalar mean_L = mean(img_L(Rect(img_L.cols - W, 0, W, img_L.rows)));
    Scalar mean_M = mean(img_M(Rect(0, 0, W, img_M.rows)));
    
    // BGR通道差值
    int delta_b = clamp(static_cast<int>(mean_L[0] - mean_M[0]), -50, 50);
    int delta_g = clamp(static_cast<int>(mean_L[1] - mean_M[1]), -50, 50);
    int delta_r = clamp(static_cast<int>(mean_L[2] - mean_M[2]), -50, 50);
    
    // 创建调整矩阵
    Mat adjustment = Mat::zeros(img_L.size(), img_L.type());
    vector<Mat> adjustment_channels;
    split(adjustment, adjustment_channels);
    
    // 设置各通道的调整值
    adjustment_channels[0].setTo(Scalar(abs(delta_b)));
    adjustment_channels[1].setTo(Scalar(abs(delta_g)));
    adjustment_channels[2].setTo(Scalar(abs(delta_r)));
    
    merge(adjustment_channels, adjustment);
    
    // 调整亮度
    Mat img_left_adjusted = img_L.clone();
    vector<Mat> adjusted_channels;
    split(img_left_adjusted, adjusted_channels);
    vector<Mat> original_channels;
    split(img_L, original_channels);
    
    std::vector<std::thread> threads(3);

    // 对每个通道分别处理
    for(int i = 0; i < 3; i++) {
        threads.emplace_back([i,delta_b,delta_g,delta_r,&original_channels,&adjustment_channels,&adjusted_channels]{

            double delta = (i == 0) ? delta_b : (i == 1) ? delta_g : delta_r;
        
        
            if(delta >= 0) {
                subtract(original_channels[i], adjustment_channels[i], adjusted_channels[i]);
            } else {
                add(original_channels[i], adjustment_channels[i], adjusted_channels[i]);
            }
        });
      
    }

    for(auto& v:threads)
    {
        if(v.joinable())
        v.join();
    }
    
    merge(adjusted_channels, img_left_adjusted);

    return img_left_adjusted;
}

Mat ColorBlance::colorblance_right(const Mat& img_M, const Mat& img_R, int W) {

    // 计算亮度差
    Scalar mean_M = mean(img_M(Rect(img_M.cols - W, 0, W, img_M.rows)));
    Scalar mean_R = mean(img_R(Rect(0, 0, W, img_R.rows)));
    
    // BGR通道差值
    int delta_b = clamp(static_cast<int>(mean_M[0] - mean_R[0]), -50, 50);
    int delta_g = clamp(static_cast<int>(mean_M[1] - mean_R[1]), -50, 50);
    int delta_r = clamp(static_cast<int>(mean_M[2] - mean_R[2]), -50, 50);
    
    // 创建调整矩阵
    Mat adjustment = Mat::zeros(img_R.size(), img_R.type());
    vector<Mat> adjustment_channels;
    split(adjustment, adjustment_channels);
    
    // 设置各通道的调整值
    adjustment_channels[0].setTo(Scalar(abs(delta_b)));
    adjustment_channels[1].setTo(Scalar(abs(delta_g)));
    adjustment_channels[2].setTo(Scalar(abs(delta_r)));
    
    merge(adjustment_channels, adjustment);
    
    // 调整亮度
    Mat img_right_adjusted = img_R.clone();
    vector<Mat> adjusted_channels;
    split(img_right_adjusted, adjusted_channels);
    vector<Mat> original_channels;
    split(img_R, original_channels);
    
    // 对每个通道分别处理
    for(int i = 0; i < 3; i++) {
        double delta = (i == 0) ? delta_b : (i == 1) ? delta_g : delta_r;
        if(delta >= 0) {
            add(original_channels[i], adjustment_channels[i], adjusted_channels[i]);
        } else {
            subtract(original_channels[i], adjustment_channels[i], adjusted_channels[i]);
        }
    }
    
    merge(adjusted_channels, img_right_adjusted);
    return img_right_adjusted;

}

void ColorBlance::color_balance_peizhun_left(const Mat& img_L, const Mat& img_M, int W, int y_s_1, int y_x_1,
                                           const Mat& H_left, Mat& left_warped, Mat& canvas) {

    // 亮度平衡 6ms
     Mat adjusted_L = colorblance_left(img_L, img_M, W);

     // 裁剪中图
     Mat img_M_cropped = img_M(Rect(0, y_s_1, img_M.cols, y_x_1 - y_s_1)).clone();

     // 创建画布并复制中图
     canvas = Mat::zeros(img_M_cropped.rows, adjusted_L.cols + img_M_cropped.cols, CV_8UC3);
     img_M_cropped.copyTo(canvas(Rect(adjusted_L.cols, 0, img_M_cropped.cols, img_M_cropped.rows)));

     // 左图透视变换 60ms
    cv::cuda::GpuMat d_adjusted_L;
    cv::cuda::GpuMat d_left_warped;
    d_adjusted_L.upload(adjusted_L);
    cv::cuda::warpPerspective(
        d_adjusted_L,    // 输入（GPU 内存）
        d_left_warped,   // 输出（GPU 内存）
        H_left,          // 变换矩阵（自动上传到 GPU）
        canvas.size(),     // 输出尺寸
        cv::INTER_LINEAR // 插值方法
    );
    d_left_warped.download(left_warped); // 数据下载到 CPU
    //  cv::cuda::warpPerspective(adjusted_L, left_warped, H_left, canvas.size());
    //  cv::warpPerspective(adjusted_L, left_warped, H_left, canvas.size());
                                            
}

void ColorBlance::color_balance_peizhun_right(const Mat& img_M, const Mat& img_R, int W, int y_s_1, int y_x_1,
                                            const Mat& H_right, Mat& canvas, Mat& right_warped) {
    // 亮度平衡 26ms
    Mat adjusted_R = colorblance_right(img_M, img_R, W);
            
    // 裁剪中图
    Mat img_M_cropped = img_M(Rect(0, y_s_1, img_M.cols, y_x_1 - y_s_1)).clone();
    
    // 右图透视变换
    cv::cuda::GpuMat d_adjusted_R;
    cv::cuda::GpuMat d_right_warped;
    d_adjusted_R.upload(adjusted_R);
    cv::cuda::warpPerspective(
        d_adjusted_R,    // 输入（GPU 内存）
        d_right_warped,   // 输出（GPU 内存）
        H_right,          // 变换矩阵（自动上传到 GPU）
        Size(adjusted_R.cols + img_M_cropped.cols, img_M_cropped.rows),     // 输出尺寸
        cv::INTER_LINEAR // 插值方法
    );
    d_right_warped.download(right_warped);

    // warpPerspective(adjusted_R, right_warped, H_right, 
    //                Size(adjusted_R.cols + img_M_cropped.cols, img_M_cropped.rows));
    
    // 创建画布并复制中图
    canvas = Mat::zeros(right_warped.size(), CV_8UC3);
    img_M_cropped.copyTo(canvas(Rect(0, 0, img_M_cropped.cols, img_M_cropped.rows)));
}