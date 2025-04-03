#include "readData.h"

// CalibrationData 结构体的析构函数
CalibrationData::~CalibrationData() {
    delete[] target_indices;
    delete[] valid_mask;
    delete[] source_coords;
    delete[] H_left;
    delete[] H_right;
    delete[] weights_left;
    delete[] weights_right;
}

// 获取权重总大小的辅助函数
size_t CalibrationData::getWeightsSize() const {
    return weights_dims[0] * weights_dims[1] * weights_dims[2];
}

// 打印矩阵的辅助函数
template <typename T>
void printMatrix(const T *data, const std::vector<size_t> &shape) {
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (size_t i = 0; i < shape[0]; ++i) {
        std::cout << "[";
        for (size_t j = 0; j < shape[1]; ++j) {
            std::cout << data[i * shape[1] + j];
            if (j < shape[1] - 1)
                std::cout << " ";
        }
        std::cout << "]" << std::endl;
    }
}

// 加载权重数据的辅助函数
void LoadWeights(cnpy::npz_t& data, CalibrationData& result) {
    //std::cout << "Loading weights..." << std::endl;
    
    // 加载 weights_left
    cnpy::NpyArray arr_weights_left = data["weights_left"];
    if (arr_weights_left.shape.size() != 3) {
        throw std::runtime_error("weights_left must be 3-dimensional");
    }
    
    for (int i = 0; i < 3; i++) {
        result.weights_dims[i] = arr_weights_left.shape[i];
    }
    
    size_t total_size = result.getWeightsSize();
    result.weights_left = new float[total_size];
    result.weights_right = new float[total_size];
    
    std::memcpy(result.weights_left, arr_weights_left.data<float>(), total_size * sizeof(float));
    
    cnpy::NpyArray arr_weights_right = data["weights_right"];
    if (arr_weights_right.shape != arr_weights_left.shape) {
        throw std::runtime_error("weights_left and weights_right must have same dimensions");
    }
    
    std::memcpy(result.weights_right, arr_weights_right.data<float>(), total_size * sizeof(float));
    
    result.weights_left_shape = arr_weights_left.shape;
    result.weights_right_shape = arr_weights_right.shape;

    // 结构化打印函数
    auto PrintWeights = [](const std::string& name, float* weights, const size_t* dims) {
        const size_t dim0 = dims[0], dim1 = dims[1], dim2 = dims[2];
        const int SHOW_DIMS = 3;  // 每个维度显示首尾3个元素
        
        std::cout << name << ":\n [";
        for (size_t i = 0; i < dim0; ++i) {
            // 维度截断逻辑
            const bool show_i = (i < SHOW_DIMS) || (i >= dim0 - SHOW_DIMS);
            if (dim0 > 2*SHOW_DIMS && !show_i) {
                if (i == SHOW_DIMS) std::cout << "\n ...";
                continue;
            }

            // 层级缩进控制
            if (i > 0) std::cout << "\n ";
            std::cout << "[";
            for (size_t j = 0; j < dim1; ++j) {
                const bool show_j = (j < SHOW_DIMS) || (j >= dim1 - SHOW_DIMS);
                if (dim1 > 2*SHOW_DIMS && !show_j) {
                    if (j == SHOW_DIMS) std::cout << "\n  ...";
                    continue;
                }

                if (j > 0) std::cout << "\n  ";
                std::cout << "[";
                for (size_t k = 0; k < dim2; ++k) {
                    const size_t index = i*dim1*dim2 + j*dim2 + k;
                    
                    // 特殊格式化：整数显示为x.，浮点数显示实际值
                    float val = weights[index];
                    if (val == static_cast<int>(val)) {
                        std::cout << static_cast<int>(val) << ". ";
                    } else {
                        std::cout << std::setprecision(6) << val << " ";
                    }
                }
                std::cout << "\b]";  // 回退多余空格
            }
            std::cout << "]";
        }
        std::cout << "]\n\n";
    };

    // PrintWeights("weights_left", result.weights_left, result.weights_dims);
    // PrintWeights("weights_right", result.weights_right, result.weights_dims);
    
    //std::cout << "Weights loaded successfully" << std::endl;
}
struct Result {
    int* target_indices;
    std::vector<size_t> target_indices_shape;
};

// 读取校准数据的函数
CalibrationData ReadCalibrationData(const std::string &filename) {
    CalibrationData result;

    try {
        //std::cout << "Loading NPZ file: " << filename << std::endl;
        cnpy::npz_t data = cnpy::npz_load(filename);

        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

        // 读取 H_left
        // std::cout << "Loading H_left..." << std::endl;
        cnpy::NpyArray arr_H_left = data["H_left"];
        size_t H_left_size = arr_H_left.shape[0] * arr_H_left.shape[1];
        result.H_left = new double[H_left_size];
        std::memcpy(result.H_left, arr_H_left.data<double>(), H_left_size * sizeof(double));
        result.H_left_shape = arr_H_left.shape;

        // 读取 H_right
        // std::cout << "Loading H_right..." << std::endl;
        cnpy::NpyArray arr_H_right = data["H_right"];
        size_t H_right_size = arr_H_right.shape[0] * arr_H_right.shape[1];
        result.H_right = new double[H_right_size];
        std::memcpy(result.H_right, arr_H_right.data<double>(), H_right_size * sizeof(double));
        result.H_right_shape = arr_H_right.shape;

        // 读取标量值
        cnpy::NpyArray arr_rows = data["rows"];
        result.rows = arr_rows.data<int>()[0];

        cnpy::NpyArray arr_output_cols = data["output_cols"];
        result.output_cols = arr_output_cols.data<int>()[0];

        // 读取 valid_mask
        // std::cout << "Loading valid_mask..." << std::endl;
        cnpy::NpyArray arr_valid_mask = data["valid_mask"];
        size_t valid_mask_size = arr_valid_mask.shape[0] * arr_valid_mask.shape[1];
        result.valid_mask = new bool[valid_mask_size];
        std::memcpy(result.valid_mask, arr_valid_mask.data<bool>(), valid_mask_size * sizeof(bool));
        result.valid_mask_shape = arr_valid_mask.shape;

        // 读取 target_indices 的正确方式
        // std::cout << "Loading target_indices..." << std::endl;
        cnpy::NpyArray arr_target_indices = data["target_indices"];

        // 验证数组维度
        if (arr_target_indices.shape.size() != 1) {
            throw std::runtime_error("target_indices must be 1-dimensional");
        }

        // 通过word_size和类型字符验证数据类型
        if (arr_target_indices.word_size != sizeof(int64_t)) {
            std::cerr << "数据类型错误: 期望8字节，实际" 
                    << arr_target_indices.word_size << "字节" << std::endl;
            throw std::runtime_error("target_indices数据类型不匹配");
        }

        // 分配内存并复制数据
        size_t total_size = arr_target_indices.shape[0];
        result.target_indices = new int64_t[total_size];  // 正确分配int64_t数组
        std::memcpy(result.target_indices, 
                arr_target_indices.data<int64_t>(),  // 使用模板特化
                total_size * sizeof(int64_t));
        result.target_indices_shape = arr_target_indices.shape;


        // 读取source_coords
        // std::cout << "Loading source_coords..." << std::endl;
        cnpy::NpyArray arr_source_coords = data["source_coords"];
        
        // 确保 source_coords 是二维数组
        if (arr_source_coords.shape.size() != 2 || arr_source_coords.shape[1] != 2) {
            throw std::runtime_error("source_coords must be a 2D array with shape (N, 2)");
        }
        
        size_t num_coords = arr_source_coords.shape[0];
        result.source_coords = new int[num_coords * 2]; // 每个坐标对有两个值 (x, y)
        std::memcpy(result.source_coords, arr_source_coords.data<int>(), num_coords * 2 * sizeof(int));
        result.source_coords_shape = arr_source_coords.shape;

        // 读取标量值
        cnpy::NpyArray arr_y_s_1 = data["y_s_1"];
        result.y_s_1 = arr_y_s_1.data<int>()[0];

        cnpy::NpyArray arr_y_x_1 = data["y_x_1"];
        result.y_x_1 = arr_y_x_1.data<int>()[0];

        cnpy::NpyArray arr_W = data["W"];
        result.W = arr_W.data<int>()[0];

        // 加载权重数据
        LoadWeights(data, result);

        // 读取 a 和 b
        cnpy::NpyArray arr_a = data["a"];
        result.a = arr_a.data<int>()[0];

        cnpy::NpyArray arr_b = data["b"];
        result.b = arr_b.data<int>()[0];

        // std::cout << "All data loaded successfully" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error loading calibration data: " << e.what() << std::endl;
        throw;
    }

    return result;
}

// 打印校准数据的函数
void PrintCalibrationData(const CalibrationData &data) {
    std::cout << "H_left:" << std::endl;
    printMatrix(data.H_left, data.H_left_shape);

    std::cout << "\nH_right:" << std::endl;
    printMatrix(data.H_right, data.H_right_shape);

    std::cout << "\nrows: " << data.rows << std::endl;
    std::cout << "output_cols: " << data.output_cols << std::endl;
    std::cout << "y_s_1: " << data.y_s_1 << std::endl;
    std::cout << "y_x_1: " << data.y_x_1 << std::endl;
    std::cout << "W: " << data.W << std::endl;
    std::cout << "a: " << data.a << std::endl;
    std::cout << "b: " << data.b << std::endl;

    std::cout << "\nArray dimensions:" << std::endl;
    std::cout << "valid_mask: " << data.valid_mask_shape[0] << "x" << data.valid_mask_shape[1] << std::endl;
    //printMatrix(data.valid_mask, data.valid_mask_shape);

    std::cout << "source_coords: " << data.source_coords_shape[0] << "x" << data.source_coords_shape[1] << std::endl;
    //printMatrix(data.source_coords, data.source_coords_shape);
    std::cout << "weights dimensions: " << data.weights_dims[0] << "x" << data.weights_dims[1] << "x" << data.weights_dims[2] << std::endl;
    
}

// 转换权重到 OpenCV Mat 的辅助函数
void convertWeightsToMat(const CalibrationData& data, cv::Mat& weightsLeft, cv::Mat& weightsRight) {
    std::vector<int> sizes = {
        static_cast<int>(data.weights_dims[0]),
        static_cast<int>(data.weights_dims[1]),
        static_cast<int>(data.weights_dims[2])
    };
    
    weightsLeft.create(3, sizes.data(), CV_64F);
    weightsRight.create(3, sizes.data(), CV_64F);
    
    std::memcpy(weightsLeft.data, data.weights_left, data.getWeightsSize() * sizeof(double));
    std::memcpy(weightsRight.data, data.weights_right, data.getWeightsSize() * sizeof(double));
}