#include "stitching.h"
#include "color_balance.h"
#include "cylinder_projection.h"
#include <vector>
#include <opencv2/core/utility.hpp>
#include <chrono>
#include <iostream>
#include "stitch.h"
#include "cylinder.h"
#include <future>
using namespace cv;
using namespace std;
using namespace std::chrono;

void stitch(const cv::Mat& img1, const cv::Mat& img2, const float* weights, cv::Mat& result) {
    cv::setNumThreads(1);
    // 输入验证
    CV_Assert(!img1.empty() && !img2.empty());
    CV_Assert(img1.type() == CV_8UC3 && img2.type() == CV_8UC3);
    // CV_Assert(img1.size() == cv::Size(2222, 584) && img1.size() == img2.size());
    CV_Assert(img1.isContinuous() && img2.isContinuous()); // 确保无padding的内存布局

    // 准备输出矩阵（584x2222 CV_8UC3）
    result.create(img1.size(), CV_8UC3);

    const int rows = img1.rows;
    const int cols = img1.cols;

    // 使用OpenCV并行框架加速处理
    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) { 
            // 遍历行
            // 获取当前行指针
            const uchar* img1_row = img1.ptr<uchar>(i);
            const uchar* img2_row = img2.ptr<uchar>(i);
            uchar* result_row = result.ptr<uchar>(i);

            // 计算当前行在权重数组中的起始偏移量
            const int weight_row_offset = i * cols * 3;

            for (int j = 0; j < cols; ++j) { // 遍历列
                // 计算当前像素在权重数组中的偏移量
                const int weight_pixel_offset = weight_row_offset + j * 3;

                // 同时处理BGR三个通道
                for (int ch = 0; ch < 3; ++ch) {
                    // 获取权重值（0.0~1.0）
                    const float w = weights[weight_pixel_offset + ch];

                    // 线性插值计算
                    const float blended = w * img1_row[j * 3 + ch] 
                                        + (1.0f - w) * img2_row[j * 3 + ch];

                    // 饱和转换到0-255范围（比static_cast更安全）
                    result_row[j * 3 + ch] = cv::saturate_cast<uchar>(blended);
                }
            }
        }
    });
}

cv::Mat process_and_stitch_images(const cv::Mat& img_left, const cv::Mat& img_mid, const cv::Mat& img_right, 
    const CalibrationData& data, const vector<Point>& source_coords,const cv::Mat& H_left,const cv::Mat& H_right,
    vector<int>& target_indices, const int& rows, const int& output_cols,const int& y_s_1, const int& y_x_1,
    const int& W,const float* weights_left, float* weights_right) { 
    Mat img_mid_left, img_mid_right, img_left_proj, img_right_proj;
    // 进行柱面投影，每次投影耗时10ms，复制图像耗时3ms，一共耗时33ms左右

    std::thread th5([&] { 
        auto start_proj = high_resolution_clock::now();
        // apply_cylinder_projection(img_mid, img_mid_left, rows, output_cols, target_indices, source_coords);
        apply_cylinder_projection(img_mid, img_mid_left, rows, output_cols, target_indices, source_coords);
        
        auto end__proj = high_resolution_clock::now();
        std::cout << "柱面投影耗时: " << duration_cast<milliseconds>(end__proj - start_proj).count() << "ms" << std::endl;
        img_mid_right = img_mid_left; 

    });

    std::thread th6([&] {
        apply_cylinder_projection(img_left, img_left_proj, rows, output_cols, target_indices, source_coords);
    });
    std::thread th7([&] {
        apply_cylinder_projection(img_right, img_right_proj, rows, output_cols, target_indices, source_coords);
    });
    th5.join();
    th6.join();
    th7.join();

    // imwrite("img_mid_left.jpg", img_mid_left);
    // imwrite("img_mid_right.jpg", img_mid_right);




    
    // 左中颜色平衡和配准
    Mat left_warped, left_canvas;
    

    auto start_color_right = high_resolution_clock::now();
    std::thread th1([&]{
        ColorBlance::color_balance_peizhun_left(img_left_proj, img_mid_left, W, y_s_1, y_x_1, H_left,left_warped, left_canvas);
    });

    Mat right_warped, right_canvas;

    std::thread th2([&]{
        ColorBlance::color_balance_peizhun_right(img_mid_right, img_right_proj, W, y_s_1, y_x_1, H_right,right_canvas, right_warped);
    });

    th1.join();
    th2.join();
    auto end_color_right = high_resolution_clock::now();
    std::cout << "右中颜色平衡和配准耗时: " << duration_cast<milliseconds>(end_color_right - start_color_right).count() << "ms" << std::endl;

    //并行执行左右拼接

    auto start_stitch = high_resolution_clock::now();


    // 并行执行左右拼接
    cv::Mat result_left, result_right;

    std::thread th3([&]{
        stitch_cuda(left_warped, left_canvas, weights_left, result_left);
        // stitch(left_warped, left_canvas, weights_left, result_left);
    });

    std::thread th4([&]{
        stitch_cuda(right_canvas, right_warped, weights_right, result_right);
        // stitch(right_canvas, right_warped, weights_right, result_right);
    });

    th3.join();
    th4.join();

    auto end_stitch = high_resolution_clock::now();
    // cpu运行55ms， gpu运行时间25ms
    std::cout << "图像拼接耗时: " << duration_cast<milliseconds>(end_stitch - start_stitch).count() << "ms" << std::endl;

    // 最终拼接和裁剪 时间忽略不计 3ms

    Mat final_img;

    // 水平拼接左右两个处理后的图像区域
    // 参数说明：
    // result_left(Rect(0, 0, data.a, result_left.rows)) : 取左半部分图像的有效区域
    //   - data.a 表示左半部分需要保留的宽度（从第0列开始截取a列）
    // result_right(Rect(data.b, 0, result_right.cols - data.b, result_right.rows)) : 取右半部分图像的有效区域
    //   - data.b 表示右半部分的起始列，去除左侧可能存在的黑边或重叠区域
    //   - result_right.cols - data.b 计算右半部分的有效宽度
    hconcat(result_left(Rect(0, 0, data.a, result_left.rows)),
    result_right(Rect(data.b, 0, result_right.cols - data.b, result_right.rows)),
    final_img);

    // Rect(500, 0, final_img.cols - 1000, final_img.rows) 参数说明：
    //   - x=500 : 从水平方向第500像素开始裁剪（去除左右各500像素的黑边）
    //   - y=0 : 垂直方向保持完整高度
    //   - width=final_img.cols-1000 : 最终保留中间区域宽度（原宽度-左右各500像素）
    //   - height=final_img.rows : 保持原始高度
    // clone() 创建独立内存拷贝，避免原图数据被后续操作修改
    Mat final_cropped = final_img(Rect(800, 0, final_img.cols - 1600, final_img.rows)).clone();

    return final_cropped;

}