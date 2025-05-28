cmake_minimum_required(VERSION 3.24)  # For CUDA_ARCHITECTURES native
project(image_segmentation LANGUAGES CXX CUDA)

# CUDA setup
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES native)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# ------------ Source Files ------------

# CPU demo source files
set(CPU_SOURCE_FILES
    src/cpu/main_cpu.cpp
    src/cpu/gaussian.cpp
    src/cpu/threshold.cpp
    src/cpu/labeling.cpp
    src/cpu/image_io.cpp
)

# GPU demo source files
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

# ------------ Executables ------------

add_executable(cpu_segmentation "${CPU_SOURCE_FILES}")
add_executable(gpu_segmentation "${GPU_SOURCE_FILES}")

# ------------ CUDA Libraries ------------

set(GPU_LINK_LIBS
    CUDA::cublas
    CUDA::cusolver
)

target_link_libraries(gpu_segmentation "${GPU_LINK_LIBS}")

# ------------ Optional Compile Flags ------------

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--keep;-src-in-ptx;--generate-line-info>)
