#include <opencv2/opencv.hpp>
#include "readData.h"
#include "stitching.h"

using namespace cv;
using namespace std;
using namespace std::chrono;

int main() {
    // 读取拼接标定数据
    auto data = ReadCalibrationData("../calibration_data.npz");
    // PrintCalibrationData (data);
    Mat img_left = imread("../now/left.jpg");
    Mat img_mid = imread("../now/mid.jpg");
    Mat img_right = imread("../now/right.jpg");
    
    // 将读取到的标定数据写为全局变量
    vector<Point> source_coords;
    for (size_t i = 0; i < data.source_coords_shape[0]; ++i) 
    {
        source_coords.emplace_back(data.source_coords[2*i], data.source_coords[2*i+1]);
    }
    Mat H_left(data.H_left_shape[0], data.H_left_shape[1], CV_64F, data.H_left);
    Mat H_right(data.H_right_shape[0], data.H_right_shape[1], CV_64F, data.H_right);
    vector<int> target_indices;
    target_indices = vector<int>(data.target_indices, data.target_indices + data.target_indices_shape[0]);
    int rows = data.rows;
    int output_cols = data.output_cols;
    int y_s_1 = data.y_s_1;
    int y_x_1 = data.y_x_1;
    int W = data.W;
    float* weights_left = data.weights_left;
    float* weights_right = data.weights_right;

    // 图像拼接
    for (int i=1;i<=20;i++){
        auto start_source_coords = high_resolution_clock::now();
        Mat result = process_and_stitch_images(img_left, img_mid, img_right, data, source_coords, H_left,H_right,target_indices,
            rows,output_cols,y_s_1,y_x_1,W, weights_left, weights_right);

        auto end_source_coords = high_resolution_clock::now();
        std::cout << "总体拼接时间: " << duration_cast<milliseconds>(end_source_coords - start_source_coords).count() << "ms" << std::endl;
        imwrite("stitched_result.jpg", result);
    }
    
    // try {
        
    //     Mat result = process_and_stitch_images(img_left, img_mid, img_right, data, source_coords, H_left,H_right,target_indices,
    //         rows,output_cols,y_s_1,y_x_1,W, weights_left, weights_right);
        
    //     imwrite("stitched_result.jpg", result);
    // } catch (const exception& e) {
    //     cerr << "Error: " << e.what() << endl;
    //     return -1;
    // }

    return -1;
}