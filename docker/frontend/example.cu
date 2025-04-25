#include <iostream>

__global__ void hello_from_gpu() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    std::cout << "Hello from CPU!" << std::endl;
    hello_from_gpu<<<1, 5>>>();  // Launch kernel with 5 threads
    cudaDeviceSynchronize();     // Wait for GPU to finish
    return 0;
}
