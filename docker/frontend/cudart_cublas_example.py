#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    const int size = 5;
    float a[size] = {1, 2, 3, 4, 5};
    float b[size] = {6, 7, 8, 9, 10};
    float c[size]; // To store the result

    float *d_a, *d_b;

    // Allocate memory on device
    cudaMalloc((void**)&d_a, size * sizeof(float));
    cudaMalloc((void**)&d_b, size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform c = a + b using cublasSaxpy (b = 1.0 * a + b)
    float alpha = 1.0f;
    cublasSaxpy(handle, size, &alpha, d_a, 1, d_b, 1);

    // Copy result to host
    cudaMemcpy(c, d_b, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result: ";
    for (int i = 0; i < size; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
