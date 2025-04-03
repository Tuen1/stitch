cmake_minimum_required(VERSION 3.10)
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/opt/cmake/share/cmake-3.31/Modules/")
# 指定C++标准
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++11")
set(CNPY_LIBRARY "/home/nvidia/ljw/VideoStitching/cnpy-master/build/libcnpy.so")
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(PROJECT_NAME stitch)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
# find_package(Thrust REQUIRED)
find_package(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(TBB REQUIRED tbb)


# cuda_add_library(cuda_lib 
#     src/imgprocess.cu
# )


# project(${PROJECT_NAME} LANGUAGES CXX CUDA)

# 包含头文件目录
include_directories(${OpenCV_INCLUDE_DIRS} include src ${CUDA_INCLUDE_DIRS} )

# （6）添加CUDA源文件
set(CUDA_SOURCE_FILES src/stitch.cu src/cylinder.cu)

cuda_add_library(cuda_lib ${CUDA_SOURCE_FILES})

add_definitions(-D_DEBUG)
# 指定源文件
add_executable(${PROJECT_NAME} 
    src/readData.cpp 
    src/color_balance.cpp 
    src/cylinder_projection.cpp 
    src/stitching.cpp
    main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CNPY_LIBRARY} 
    cnpy 
    Threads::Threads
    cuda_lib
    ${CUDA_LIBRARIES} )

