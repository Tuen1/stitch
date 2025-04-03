#ifndef READ_DATA_H
#define READ_DATA_H

#include <cnpy.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// CalibrationData 结构体声明
struct CalibrationData {
    double *H_left = nullptr;
    double *H_right = nullptr;
    int rows;
    int output_cols;
    bool *valid_mask = nullptr;
    int64_t* target_indices = nullptr;
    int *source_coords = nullptr;
    int y_s_1;
    int y_x_1;
    int W;
    float *weights_left = nullptr;
    float *weights_right = nullptr;
    int a;
    int b;

    std::vector<size_t> H_left_shape;
    std::vector<size_t> H_right_shape;
    std::vector<size_t> valid_mask_shape;
    std::vector<size_t> target_indices_shape;
    std::vector<size_t> source_coords_shape;
    std::vector<size_t> weights_left_shape;
    std::vector<size_t> weights_right_shape;

    size_t weights_dims[3];

    size_t getWeightsSize() const;
    ~CalibrationData();
};

// 打印矩阵的辅助函数
template <typename T>
void printMatrix(const T *data, const std::vector<size_t> &shape);

// 加载权重数据的辅助函数
void LoadWeights(cnpy::npz_t& data, CalibrationData& result);

// 读取校准数据的函数
CalibrationData ReadCalibrationData(const std::string &filename);

// 打印校准数据的函数
void PrintCalibrationData(const CalibrationData &data);

// 转换权重到 OpenCV Mat 的辅助函数
void convertWeightsToMat(const CalibrationData& data, cv::Mat& weightsLeft, cv::Mat& weightsRight);

#endif // READ_DATA_H