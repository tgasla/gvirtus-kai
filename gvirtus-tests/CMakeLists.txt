cmake_minimum_required(VERSION 3.17)
project(test_cuda_libs LANGUAGES CXX CUDA)

cmake_policy(SET CMP0104 NEW)

# Set a default CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()

# Require C++17 (optional, but good practice)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find GoogleTest package (make sure it is installed or handled otherwise)
find_package(GTest REQUIRED)

# Your CUDA source file
set(SOURCES
    test_cudart.cu
    test_cublas.cu
    test_curand.cu
    test_cufft.cu
)

# Add executable target
add_executable(test_cuda_libs ${SOURCES})

set_target_properties(test_cuda_libs PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
)

# Include directories (GoogleTest and GVirtuS)
target_include_directories(test_cuda_libs PRIVATE
    ${GTEST_INCLUDE_DIRS}
    $ENV{GVIRTUS_HOME}/include
)

# Link GVirtuS libs
target_link_directories(test_cuda_libs PRIVATE
    $ENV{GVIRTUS_HOME}/lib/frontend
)

# Link GoogleTest libraries
target_link_libraries(test_cuda_libs PRIVATE
    GTest::GTest
    GTest::Main
    curand
    cudart
    cublas
    cufft
    cudnn
)

set_target_properties(test_cuda_libs PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)