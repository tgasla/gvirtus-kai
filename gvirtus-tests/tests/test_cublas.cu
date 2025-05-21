#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(err) ASSERT_EQ((err), cudaSuccess)
#define CUBLAS_CHECK(err) ASSERT_EQ((err), CUBLAS_STATUS_SUCCESS)

TEST(cuBLAS, CreateDestroy) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasDestroy(handle));
}

TEST(cuBLAS, SetStreamDestroy) {
    cublasHandle_t handle;
    cudaStream_t stream;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(handle));
}

TEST(cuBLAS, Sgemm) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Matrix size 2x2 for simplicity
    const int N = 2;
    float h_A[] = {1, 2, 3, 4};  // col-major 2x2
    float h_B[] = {5, 6, 7, 8};
    float h_C[4] = {0};

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, N * N * sizeof(float)));

    float alpha = 1.0f, beta = 0.0f;

    // SGEMM: C = alpha * A * B + beta * C
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             d_A, N,
                             d_B, N,
                             &beta,
                             d_C, N));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(h_C), cudaMemcpyDeviceToHost));

    // Check a few expected values
    ASSERT_FLOAT_EQ(h_C[0], 19.0f); // 1*5+3*6=5+18=23 (wait: col-major, careful!)
    // Let's calculate correct expected values for col-major:
    // C = A * B, with A and B col-major:
    // A = |1 3|
    //     |2 4|
    // B = |5 7|
    //     |6 8|
    // C[0,0] = 1*5 + 3*6 = 5 + 18 = 23
    // C[1,0] = 2*5 + 4*6 = 10 + 24 = 34
    // C[0,1] = 1*7 + 3*8 = 7 + 24 = 31
    // C[1,1] = 2*7 + 4*8 = 14 + 32 = 46

    ASSERT_FLOAT_EQ(h_C[0], 23.0f);
    ASSERT_FLOAT_EQ(h_C[1], 34.0f);
    ASSERT_FLOAT_EQ(h_C[2], 31.0f);
    ASSERT_FLOAT_EQ(h_C[3], 46.0f);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));
}

TEST(cuBLAS, Sgemv) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const int M = 2, N = 2;
    float h_A[] = {1, 3, 2, 4};  // 2x2 col-major
    float h_x[] = {1, 2};
    float h_y[] = {0, 0};

    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, M * sizeof(float)));

    float alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasSgemv(handle,
                             CUBLAS_OP_N,
                             M, N,
                             &alpha,
                             d_A, M,
                             d_x, 1,
                             &beta,
                             d_y, 1));

    CUDA_CHECK(cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost));

    // y = A*x = [1*1+3*2, 2*1+4*2] = [1+6, 2+8] = [7, 10]
    ASSERT_FLOAT_EQ(h_y[0], 7.0f);
    ASSERT_FLOAT_EQ(h_y[1], 10.0f);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUBLAS_CHECK(cublasDestroy(handle));
}

TEST(cuBLAS, Saxpy) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int n = 3;
    float h_x[] = {1, 2, 3};
    float h_y[] = {4, 5, 6};

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, sizeof(h_y), cudaMemcpyHostToDevice));

    float alpha = 2.0f;

    CUBLAS_CHECK(cublasSaxpy(handle, n, &alpha, d_x, 1, d_y, 1));

    CUDA_CHECK(cudaMemcpy(h_y, d_y, sizeof(h_y), cudaMemcpyDeviceToHost));

    // y = y + alpha*x = [4+2*1,5+2*2,6+2*3] = [6,9,12]
    ASSERT_FLOAT_EQ(h_y[0], 6.0f);
    ASSERT_FLOAT_EQ(h_y[1], 9.0f);
    ASSERT_FLOAT_EQ(h_y[2], 12.0f);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUBLAS_CHECK(cublasDestroy(handle));
}

TEST(cuBLAS, Scopy) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int n = 3;
    float h_x[] = {1, 2, 3};
    float h_y[] = {0, 0, 0};

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, sizeof(h_y), cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasScopy(handle, n, d_x, 1, d_y, 1));

    CUDA_CHECK(cudaMemcpy(h_y, d_y, sizeof(h_y), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        ASSERT_FLOAT_EQ(h_y[i], h_x[i]);
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUBLAS_CHECK(cublasDestroy(handle));
}

TEST(cuBLAS, Snrm2) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int n = 3;
    float h_x[] = {3, 4, 0};

    float *d_x;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice));

    float result = 0;
    CUBLAS_CHECK(cublasSnrm2(handle, n, d_x, 1, &result));

    ASSERT_NEAR(result, 5.0f, 1e-5);

    CUDA_CHECK(cudaFree(d_x));
    CUBLAS_CHECK(cublasDestroy(handle));
}

TEST(cuBLAS, Sdot) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int n = 3;
    float h_x[] = {1, 2, 3};
    float h_y[] = {4, 5, 6};

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, sizeof(h_y), cudaMemcpyHostToDevice));

    float result = 0;
    CUBLAS_CHECK(cublasSdot(handle, n, d_x, 1, d_y, 1, &result));

    // dot product = 1*4 + 2*5 + 3*6 = 32
    ASSERT_FLOAT_EQ(result, 32.0f);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUBLAS_CHECK(cublasDestroy(handle));
}

TEST(cuBLAS, Dgemm) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int m = 2, n = 3, k = 2;
    double alpha = 1.0, beta = 0.0;

    // A is m x k
    double h_A[] = {1.0, 2.0,
                    3.0, 4.0};

    // B is k x n
    double h_B[] = {5.0, 6.0, 7.0,
                    8.0, 9.0, 10.0};

    // C is m x n
    double h_C[6] = {0};

    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, sizeof(h_C), cudaMemcpyHostToDevice));

    // Perform C = alpha * A * B + beta * C
    // Note: cuBLAS is column-major by default, so we can either transpose matrices or switch order:
    // Using row-major layout, specify operation flags as CUBLAS_OP_T to transpose inputs

    // Here, to keep it simple, we use the matrices as column-major:
    CUBLAS_CHECK(cublasDgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                             &alpha,
                             d_A, m,
                             d_B, k,
                             &beta,
                             d_C, m));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(h_C), cudaMemcpyDeviceToHost));

    // Expected result:
    // C = [1*5+2*8, 1*6+2*9, 1*7+2*10
    //      3*5+4*8, 3*6+4*9, 3*7+4*10]
    // = [21, 24, 27
    //    47, 54, 61]

    double expected[] = {21, 24, 27, 47, 54, 61};
    for (int i = 0; i < m * n; ++i) {
        ASSERT_NEAR(h_C[i], expected[i], 1e-9);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));
}

TEST(cuBLAS, Dgemv) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int m = 2, n = 3;
    double alpha = 1.0, beta = 0.0;

    // A is m x n
    double h_A[] = {1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0};

    double h_x[] = {1.0, 1.0, 1.0};
    double h_y[] = {0.0, 0.0};

    double *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, m * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, m * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, sizeof(h_y), cudaMemcpyHostToDevice));

    // y = alpha * A * x + beta * y
    CUBLAS_CHECK(cublasDgemv(handle,
                             CUBLAS_OP_N,
                             m, n,
                             &alpha,
                             d_A, m,
                             d_x, 1,
                             &beta,
                             d_y, 1));

    CUDA_CHECK(cudaMemcpy(h_y, d_y, sizeof(h_y), cudaMemcpyDeviceToHost));

    // Expected result:
    // y[0] = 1*1 + 2*1 + 3*1 = 6
    // y[1] = 4*1 + 5*1 + 6*1 = 15

    ASSERT_NEAR(h_y[0], 6.0, 1e-9);
    ASSERT_NEAR(h_y[1], 15.0, 1e-9);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUBLAS_CHECK(cublasDestroy(handle));
}

TEST(CuBLAS, Daxpy) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int n = 3;
    double alpha = 2.0;
    double h_x[] = {1.0, 2.0, 3.0};
    double h_y[] = {4.0, 5.0, 6.0};

    double *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, sizeof(h_y), cudaMemcpyHostToDevice));

    // y = alpha * x + y
    CUBLAS_CHECK(cublasDaxpy(handle, n, &alpha, d_x, 1, d_y, 1));

    CUDA_CHECK(cudaMemcpy(h_y, d_y, sizeof(h_y), cudaMemcpyDeviceToHost));

    // Expected y = [4+2*1, 5+2*2, 6+2*3] = [6, 9, 12]
    ASSERT_NEAR(h_y[0], 6.0, 1e-9);
    ASSERT_NEAR(h_y[1], 9.0, 1e-9);
    ASSERT_NEAR(h_y[2], 12.0, 1e-9);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUBLAS_CHECK(cublasDestroy(handle));
}

TEST(cuBLAS, Dcopy) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int n = 3;
    double h_x[] = {1.0, 2.0, 3.0};
    double h_y[] = {0.0, 0.0, 0.0};

    double *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, sizeof(h_y), cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasDcopy(handle, n, d_x, 1, d_y, 1));

    CUDA_CHECK(cudaMemcpy(h_y, d_y, sizeof(h_y), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        ASSERT_DOUBLE_EQ(h_y[i], h_x[i]);
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUBLAS_CHECK(cublasDestroy(handle));
}

TEST(cuBLAS, Dnrm2) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int n = 3;
    double h_x[] = {3.0, 4.0, 0.0};

    double *d_x;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice));

    double result = 0;
    CUBLAS_CHECK(cublasDnrm2(handle, n, d_x, 1, &result));

    ASSERT_NEAR(result, 5.0, 1e-9);

    CUDA_CHECK(cudaFree(d_x));
    CUBLAS_CHECK(cublasDestroy(handle));
}

TEST(cuBLAS, Ddot) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int n = 3;
    double h_x[] = {1.0, 2.0, 3.0};
    double h_y[] = {4.0, 5.0, 6.0};

    double *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, sizeof(h_y), cudaMemcpyHostToDevice));

    double result = 0;
    CUBLAS_CHECK(cublasDdot(handle, n, d_x, 1, d_y, 1, &result));

    // dot product = 1*4 + 2*5 + 3*6 = 32
    ASSERT_DOUBLE_EQ(result, 32.0);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUBLAS_CHECK(cublasDestroy(handle));
}
