#include <iostream>
#include <cuda_runtime.h>

int main() {
    // ======== CUDA Runtime ========
    float *d_data;
    float *h_data = new float[10];  // Host memory to store the result

    // Allocate memory on the device (GPU)
    cudaError_t err = cudaMalloc(&d_data, 10 * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Set memory on the device to zero
    err = cudaMemset(d_data, 0, 10 * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA memset failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        return -1;
    }
    std::cout << "CUDA: Allocated and zeroed memory.\n";

    // Copy memory from device to host
    err = cudaMemcpy(h_data, d_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        return -1;
    }

    // Print the result to verify it's zeroed out
    std::cout << "CUDA: Copied data from device to host. Values are:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << "\n";

    // Free the allocated memory
    cudaFree(d_data);
    delete[] h_data;

    return 0;
}
