#include <iostream>
#include <cublas_v2.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <cudnn.h>

int main() {
    // ======== CUDA Runtime ========
    float *d_data;
    cudaMalloc(&d_data, 10 * sizeof(float));
    cudaMemset(d_data, 0, 10 * sizeof(float));
    std::cout << "cudart: Allocated and zeroed memory.\n";

    // ======== cuBLAS ========
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    float alpha = 1.0f;
    float *x, *y;
    cudaMalloc(&x, 5 * sizeof(float));
    cudaMalloc(&y, 5 * sizeof(float));
    cublasSaxpy(cublasHandle, 5, &alpha, x, 1, y, 1);
    std::cout << "cuBLAS: Performed SAXPY.\n";
    cublasDestroy(cublasHandle);

    // ======== cuFFT ========
    cufftHandle fftPlan;
    cufftComplex *fftData;
    cudaMalloc(&fftData, 8 * sizeof(cufftComplex));
    cufftPlan1d(&fftPlan, 8, CUFFT_C2C, 1);
    cufftExecC2C(fftPlan, fftData, fftData, CUFFT_FORWARD);
    std::cout << "cuFFT: Executed 1D FFT.\n";
    cufftDestroy(fftPlan);

    // ======== cuRAND ========
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, d_data, 10);
    std::cout << "cuRAND: Generated uniform random numbers.\n";
    //curandDestroyGenerator(gen);

    // ======== cuDNN ========
    //cudnnHandle_t cudnnHandle;
    //cudnnCreate(&cudnnHandle);
    //std::cout << "created handle\n";
    //cudnnTensorDescriptor_t desc;
    //cudnnCreateTensorDescriptor(&desc);
    //std::cout << "created tensor descriptor\n";
    //cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);
    //std::cout << "cuDNN: Created tensor descriptor.\n";
    //cudnnDestroyTensorDescriptor(desc);
    //cudnnDestroy(cudnnHandle);


    // ======== Cleanup ========
    //cudaFree(d_data);
    //cudaFree(x);
    //cudaFree(y);
    //cudaFree(fftData);

    return 0;
}
