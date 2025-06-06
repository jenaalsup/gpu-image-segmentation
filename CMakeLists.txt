cmake_minimum_required(VERSION 3.24)  # for CUDA architecture native
project(image_segmentation LANGUAGES CXX CUDA)

# enable C++20 for both C++ and CUDA
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# opencv for image io
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES native)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# cpu
set(CPU_SOURCE_FILES
    src/cpu/main_cpu.cpp
    src/cpu/gaussian.cpp
    src/cpu/threshold.cpp
    src/cpu/labeling.cpp
    src/cpu/image_io.cpp
)

# gpu
set(GPU_SOURCE_FILES
    src/gpu/main_gpu.cu
    src/gpu/gaussian.cu
    src/gpu/threshold.cu
    src/gpu/labeling.cu
    src/gpu/ErrorCheck.cuh
    src/gpu/gaussian.cuh
    src/gpu/threshold.cuh
    src/gpu/labeling.cuh
)


add_executable(cpu_segmentation "${CPU_SOURCE_FILES}")
add_executable(gpu_segmentation "${GPU_SOURCE_FILES}")
target_link_libraries(cpu_segmentation ${OpenCV_LIBS})


set(GPU_LINK_LIBS
    CUDA::cublas
    CUDA::cusolver
)

target_link_libraries(gpu_segmentation ${OpenCV_LIBS} ${GPU_LINK_LIBS})


# tests
set(TEST_SOURCE_FILES
    src/tests/test_main.cpp
    src/cpu/gaussian.cpp
    src/cpu/threshold.cpp
    src/cpu/labeling.cpp
    src/cpu/image_io.cpp
)
add_executable(cpu_tests "${TEST_SOURCE_FILES}")
target_link_libraries(cpu_tests ${OpenCV_LIBS})
