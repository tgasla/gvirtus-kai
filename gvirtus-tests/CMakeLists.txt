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
set(SOURCES test_curand.cu)

# Add executable target
add_executable(test_cuda_libs ${SOURCES})

# Pass nvcc options: shared CUDA runtime, debugging info if you want
# target_compile_options(test_cuda_libs PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:--cudart=shared> -L $ENV{GVIRTUS_HOME}/lib/frontend
# )

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
    # $ENV{GVIRTUS_HOME}/lib
)

# Link GoogleTest libraries
target_link_libraries(test_cuda_libs PRIVATE
    GTest::GTest
    GTest::Main
    # dl
    # pthread
    curand
    cudart
    cublas
    cufft
    cudnn
)

# Set RPATH so your executable finds GVirtuS shared libs at runtime
set_target_properties(test_cuda_libs PROPERTIES
#     BUILD_RPATH "$ENV{LD_LIBRARY_PATH}"
    CUDA_SEPARABLE_COMPILATION ON
)