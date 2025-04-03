#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core.hpp>
#include "stitch.h"  // 确保头文件路径正确
#include "sstream"
#include <vector>
using namespace cv;

//------------------------------------------------------------------------------------------
// CUDA核函数实现
// 注意：必须添加 __global__ 修饰符并实现具体逻辑
//------------------------------------------------------------------------------------------
extern "C" __global__ void blendImagesKernel_Optimized(
    const uchar* img1, 
    const uchar* img2, 
    const float* weights,
    uchar* output, 
    int rows, 
    int cols
) {
    // 计算当前线程的像素坐标
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        // 计算一维索引（假设图像为连续存储）
        const int idx = (y * cols + x) * 3;

        // 加权融合每个通道
        for (int c = 0; c < 3; ++c) {
            float blended = img1[idx + c] * weights[idx + c] + 
                            img2[idx + c] * (1.0f - weights[idx + c]);
            output[idx + c] = static_cast<uchar>(blended);
        }
    }
}

//------------------------------------------------------------------------------------------
// 主机函数 stitch
//------------------------------------------------------------------------------------------
void stitch_cuda(const cv::Mat& img1, const cv::Mat& img2, const float* weights, cv::Mat& result) {
    // 输入验证
    CV_Assert(!img1.empty() && !img2.empty());
    CV_Assert(img1.type() == CV_8UC3 && img2.type() == CV_8UC3);
    CV_Assert(img1.isContinuous() && img2.isContinuous());
    CV_Assert(img1.size() == img2.size());

    const int rows = img1.rows;
    const int cols = img1.cols;
    const size_t img_size = rows * cols * 3 * sizeof(uchar);
    const size_t weight_size = rows * cols * 3 * sizeof(float);

    // 设备内存指针
    uchar *d_img1 = nullptr, *d_img2 = nullptr, *d_result = nullptr;
    float *d_weights = nullptr;
    cudaStream_t stream = nullptr;

    // 初始化CUDA流
    cudaStreamCreate(&stream);

    try {
        // 分配设备内存
        cudaMalloc(&d_img1, img_size);
        cudaMalloc(&d_img2, img_size);
        cudaMalloc(&d_weights, weight_size);
        cudaMalloc(&d_result, img_size);

        // 异步拷贝数据到设备
        cudaMemcpyAsync(d_img1, img1.data, img_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_img2, img2.data, img_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_weights, weights, weight_size, cudaMemcpyHostToDevice, stream);

        // 设置核函数的执行配置
        dim3 block(32, 8);  // 每个块16x16线程
        dim3 grid((cols + block.x - 1) / block.x, 
                  (rows + block.y - 1) / block.y);

        // 调用核函数（注意：核函数名后必须使用<<<grid, block>>>语法）
        blendImagesKernel_Optimized<<<grid, block, 0, stream>>>(
            d_img1, d_img2, d_weights, d_result, rows, cols
        );

        // 准备输出矩阵
        result.create(img1.size(), CV_8UC3);

        // 异步拷贝结果回主机
        cudaMemcpyAsync(result.data, d_result, img_size, cudaMemcpyDeviceToHost, stream);

        // 同步流，确保所有操作完成
        cudaStreamSynchronize(stream);

    } catch (...) {
        // 异常时释放资源
        cudaFree(d_img1);
        cudaFree(d_img2);
        cudaFree(d_weights);
        cudaFree(d_result);
        cudaStreamDestroy(stream);
        throw;
    }

    // 释放设备内存
    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_weights);
    cudaFree(d_result);
    cudaStreamDestroy(stream);
}
