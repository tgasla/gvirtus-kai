#include <gtest/gtest.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) ASSERT_EQ((err), cudaSuccess)

TEST(cudaRT, MallocFree) {
    void* devPtr = nullptr;
    CUDA_CHECK(cudaMalloc(&devPtr, 1024));
    CUDA_CHECK(cudaFree(devPtr));
}

TEST(cudaRT, MemcpySync) {
    int h_src = 42;
    int h_dst = 0;
    int* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_ptr, &h_src, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&h_dst, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_dst, 42);

    CUDA_CHECK(cudaFree(d_ptr));
}

TEST(cudaRT, MemcpyAsync) {
    int h_src = 24;
    int h_dst = 0;
    int* d_ptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(int)));
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpyAsync(d_ptr, &h_src, sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_dst, d_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(h_dst, 24);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_ptr));
}

TEST(cudaRT, Memset) {
    int* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ptr, 0, sizeof(int)));

    int h_val = 1;
    CUDA_CHECK(cudaMemcpy(&h_val, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_val, 0);

    CUDA_CHECK(cudaFree(d_ptr));
}

TEST(cudaRT, StreamCreateDestroySynchronize) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(cudaRT, GetDevice) {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
}

TEST(cudaRT, SetDevice) {
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
}

TEST(cudaRT, DeviceSynchronize) {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// TEST(cudaRT, EventCreateRecordSynchronizeElapsedTime) {
//     cudaEvent_t start, stop;
//     CUDA_CHECK(cudaEventCreate(&start));
//     CUDA_CHECK(cudaEventCreate(&stop));

//     CUDA_CHECK(cudaEventRecord(start));
//     CUDA_CHECK(cudaEventRecord(stop));

//     CUDA_CHECK(cudaEventSynchronize(stop));

//     float elapsed_ms = 0;
//     CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
//     EXPECT_GE(elapsed_ms, 0.0f);

//     CUDA_CHECK(cudaEventDestroy(start));
//     CUDA_CHECK(cudaEventDestroy(stop));
// }
