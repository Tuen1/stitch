#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda_runtime.h>

// CUDA kernel for cylinder projection
__global__ void cylinderProjectionKernel(
    const uchar3* input, 
    uchar3* output, 
    int input_rows, 
    int input_cols, 
    int output_rows, 
    int output_cols,
    const int* target_indices,
    const int2* source_coords,
    int total_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_points)
        return;
    
    int k = target_indices[idx];
    int row = k / output_cols;
    int col = k % output_cols;
    
    if (row < 0 || row >= output_rows || col < 0 || col >= output_cols)
        return;
    
    int2 pt = source_coords[idx];
    int src_x = pt.x;
    int src_y = pt.y;
    
    if (src_x < 0 || src_x >= input_cols || src_y < 0 || src_y >= input_rows)
        return;
    
    // Calculate source and destination positions
    int src_idx = src_y * input_cols + src_x;
    int dst_idx = row * output_cols + col;
    
    // Copy the pixel value
    output[dst_idx] = input[src_idx];
}

void apply_cylinder_projection_cuda(
    const cv::Mat& img,
    cv::Mat& output,
    int rows,
    int output_cols,
    const std::vector<int>& target_indices,
    const std::vector<cv::Point>& source_coords)
{
    // Create CUDA device arrays
    cv::cuda::GpuMat d_img;
    cv::cuda::GpuMat d_output(rows, output_cols, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // Upload image to GPU
    d_img.upload(img);
    
    int total_points = target_indices.size();
    
    // Create device arrays for indices and coordinates
    int* d_target_indices = nullptr;
    int2* d_source_coords = nullptr;
    
    // Allocate memory on GPU
    cudaMalloc(&d_target_indices, total_points * sizeof(int));
    cudaMalloc(&d_source_coords, total_points * sizeof(int2));
    
    // Create host array for source coordinates in the correct format
    std::vector<int2> h_source_coords(total_points);
    for (int i = 0; i < total_points; i++) {
        h_source_coords[i].x = source_coords[i].x;
        h_source_coords[i].y = source_coords[i].y;
    }
    
    // Copy data to GPU
    cudaMemcpy(d_target_indices, target_indices.data(), total_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_coords, h_source_coords.data(), total_points * sizeof(int2), cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    int blockSize = 256;
    int numBlocks = (total_points + blockSize - 1) / blockSize;
    
    // Launch kernel
    cylinderProjectionKernel<<<numBlocks, blockSize>>>(
        (uchar3*)d_img.data,
        (uchar3*)d_output.data,
        img.rows,
        img.cols,
        rows,
        output_cols,
        d_target_indices,
        d_source_coords,
        total_points
    );
    
    // Synchronize to ensure kernel execution completes
    cudaDeviceSynchronize();
    
    // Download result from GPU
    d_output.download(output);
    
    // Free GPU memory
    cudaFree(d_target_indices);
    cudaFree(d_source_coords);
}