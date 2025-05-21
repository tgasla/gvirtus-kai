#include <gtest/gtest.h>
#include <iostream>
#include <curand.h>

TEST(cuRAND, CreateDestroyGenerator) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
}

TEST(cuRAND, CreateDestroyGeneratorHost) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
}

TEST(cuRAND, SetSeed) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
}

TEST(cuRAND, GenerateDevice) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL), CURAND_STATUS_SUCCESS);

    const size_t n = 10;
    unsigned int* output;
    ASSERT_EQ(cudaMalloc(&output, n * sizeof(unsigned int)), cudaSuccess);

    ASSERT_EQ(curandGenerate(generator, output, n), CURAND_STATUS_SUCCESS);

    unsigned int host_output[n];
    ASSERT_EQ(cudaMemcpy(host_output, output, n * sizeof(unsigned int), cudaMemcpyDeviceToHost), cudaSuccess);

    bool all_zero = true;
    for (size_t i = 0; i < n; ++i) {
        if (host_output[i] != 0) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero); // Generated numbers should not all be zero

    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(cudaFree(output), cudaSuccess);
}

TEST(cuRAND, GenerateHost) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL), CURAND_STATUS_SUCCESS);

    const size_t n = 10;
    unsigned int* output = (unsigned int*)malloc(n * sizeof(unsigned int));
    ASSERT_NE(output, nullptr);

    ASSERT_EQ(curandGenerate(generator, output, n), CURAND_STATUS_SUCCESS);

    bool all_zero = true;
    for (size_t i = 0; i < n; ++i) {
        if (output[i] != 0) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero); // Generated numbers should not all be zero

    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
    free(output);
}

TEST(cuRAND, GenerateLongLongDevice) {
    curandGenerator_t generator;
    const size_t num = 10;

    // Create a quasi-random number generator
    ASSERT_EQ(curandCreateGenerator(&generator, CURAND_RNG_QUASI_SOBOL64), CURAND_STATUS_SUCCESS) << "Failed to create generator";

    // Set dimensions (required for quasi generators)
    ASSERT_EQ(curandSetQuasiRandomGeneratorDimensions(generator, 1), CURAND_STATUS_SUCCESS) << "Failed to set dimensions";

    // Allocate device memory
    unsigned long long* d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_output, num * sizeof(unsigned long long)), cudaSuccess) << "Failed to allocate device memory";

    // Generate quasi-random numbers
    ASSERT_EQ(curandGenerateLongLong(generator, d_output, num), CURAND_STATUS_SUCCESS) << "curandGenerateLongLong failed";

    // Copy results back to host for checking
    unsigned long long h_output[num];
    ASSERT_EQ(cudaMemcpy(h_output, d_output, num * sizeof(unsigned long long), cudaMemcpyDeviceToHost), cudaSuccess) << "Failed to copy data from device to host";

    // Clean up
    ASSERT_EQ(cudaFree(d_output), cudaSuccess) << "Failed to free device memory";
    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS) << "Failed to destroy generator";
}

TEST(cuRAND, GenerateLongLongHost) {
    curandGenerator_t generator;
    const size_t num = 10;

    // Create a QUASI-random number generator (host generator)
    ASSERT_EQ(curandCreateGeneratorHost(&generator, CURAND_RNG_QUASI_SOBOL64), CURAND_STATUS_SUCCESS) << "Failed to create QUASI generator";

     // Set dimensions (required for quasi generators)
    ASSERT_EQ(curandSetQuasiRandomGeneratorDimensions(generator, 1), CURAND_STATUS_SUCCESS) << "Failed to set dimensions";

    // Allocate host memory for output
    unsigned long long* h_output = new unsigned long long[num];

    // Generate random numbers on host
    ASSERT_EQ(curandGenerateLongLong(generator, h_output, num), CURAND_STATUS_SUCCESS) << "curandGenerateLongLong failed";

    delete[] h_output;
    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS) << "Failed to destroy generator";
}

TEST(cuRAND, GenerateUniformDevice) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL), CURAND_STATUS_SUCCESS);

    const size_t n = 10;
    float* output;
    ASSERT_EQ(cudaMalloc(&output, n * sizeof(float)), cudaSuccess);

    ASSERT_EQ(curandGenerateUniform(generator, output, n), CURAND_STATUS_SUCCESS);

    float host_output[n];
    ASSERT_EQ(cudaMemcpy(host_output, output, n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_GE(host_output[i], 0.0f);
        EXPECT_LT(host_output[i], 1.0f);
    }

    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(cudaFree(output), cudaSuccess);
}

TEST(cuRAND, GenerateUniformHost) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL), CURAND_STATUS_SUCCESS);

    const size_t n = 10;
    float* output = (float*)malloc(n * sizeof(float));
    ASSERT_NE(output, nullptr);

    ASSERT_EQ(curandGenerateUniform(generator, output, n), CURAND_STATUS_SUCCESS);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_GE(output[i], 0.0f);
        EXPECT_LT(output[i], 1.0f);
    }

    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
    free(output);
}

TEST(cuRAND, GenerateNormalDevice) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 9012ULL), CURAND_STATUS_SUCCESS);

    const size_t n = 1000;  // Larger sample for stats
    float* output;
    ASSERT_EQ(cudaMalloc(&output, n * sizeof(float)), cudaSuccess);

    const float mean = 5.0f;
    const float stddev = 2.0f;
    ASSERT_EQ(curandGenerateNormal(generator, output, n, mean, stddev), CURAND_STATUS_SUCCESS);

    float host_output[n];
    ASSERT_EQ(cudaMemcpy(host_output, output, n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    // Basic sanity: check mean and stddev roughly
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += host_output[i];
    }
    float sample_mean = sum / n;
    ASSERT_NEAR(sample_mean, mean, 0.2f);

    float variance_sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = host_output[i] - mean;
        variance_sum += diff * diff;
    }
    float sample_stddev = sqrt(variance_sum / n);
    ASSERT_NEAR(sample_stddev, stddev, 0.3f);

    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(cudaFree(output), cudaSuccess);
}

TEST(cuRAND, GenerateNormalHost) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 9012ULL), CURAND_STATUS_SUCCESS);

    const size_t n = 1000;
    float* output = (float*)malloc(n * sizeof(float));
    ASSERT_NE(output, nullptr);

    const float mean = 5.0f;
    const float stddev = 2.0f;
    ASSERT_EQ(curandGenerateNormal(generator, output, n, mean, stddev), CURAND_STATUS_SUCCESS);

    // Basic sanity: check mean and stddev roughly
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += output[i];
    }
    float sample_mean = sum / n;
    ASSERT_NEAR(sample_mean, mean, 0.2f);

    float variance_sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = output[i] - mean;
        variance_sum += diff * diff;
    }
    float sample_stddev = sqrt(variance_sum / n);
    ASSERT_NEAR(sample_stddev, stddev, 0.3f);

    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
    free(output);
}

TEST(cuRAND, GenerateLogNormalDevice) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 8642ULL), CURAND_STATUS_SUCCESS);

    const size_t n = 1000;
    float* output;
    ASSERT_EQ(cudaMalloc(&output, n * sizeof(float)), cudaSuccess);

    const float mean = 0.0f;
    const float stddev = 0.5f;
    ASSERT_EQ(curandGenerateLogNormal(generator, output, n, mean, stddev), CURAND_STATUS_SUCCESS);

    float host_output[n];
    ASSERT_EQ(cudaMemcpy(host_output, output, n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    // All values should be positive
    for (size_t i = 0; i < n; ++i) {
        EXPECT_GT(host_output[i], 0.0f);
    }

    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(cudaFree(output), cudaSuccess);
}

TEST(cuRAND, GenerateLogNormalHost) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 8642ULL), CURAND_STATUS_SUCCESS);

    const size_t n = 1000;
    float* output = (float*)malloc(n * sizeof(float));
    ASSERT_NE(output, nullptr);

    const float mean = 0.0f;
    const float stddev = 0.5f;
    ASSERT_EQ(curandGenerateLogNormal(generator, output, n, mean, stddev), CURAND_STATUS_SUCCESS);

    // All values should be positive
    for (size_t i = 0; i < n; ++i) {
        EXPECT_GT(output[i], 0.0f);
    }

    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
    free(output);
}

TEST(cuRAND, GeneratePoissonDevice) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 3456ULL), CURAND_STATUS_SUCCESS);

    const size_t n = 1000;
    unsigned int* output;
    ASSERT_EQ(cudaMalloc(&output, n * sizeof(unsigned int)), cudaSuccess);

    double lambda = 4.5;
    ASSERT_EQ(curandGeneratePoisson(generator, output, n, lambda), CURAND_STATUS_SUCCESS);

    unsigned int host_output[n];
    ASSERT_EQ(cudaMemcpy(host_output, output, n * sizeof(unsigned int), cudaMemcpyDeviceToHost), cudaSuccess);

    // Basic sanity checks: values >= 0 and mean close to lambda
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        EXPECT_GE(host_output[i], 0u);
        sum += host_output[i];
    }
    double sample_mean = sum / n;
    ASSERT_NEAR(sample_mean, lambda, 0.2);

    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(cudaFree(output), cudaSuccess);
}

TEST(cuRAND, GeneratePoissonHost) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 3456ULL), CURAND_STATUS_SUCCESS);

    const size_t n = 1000;
    unsigned int* output = (unsigned int*)malloc(n * sizeof(unsigned int));
    ASSERT_NE(output, nullptr);

    double lambda = 4.5;
    ASSERT_EQ(curandGeneratePoisson(generator, output, n, lambda), CURAND_STATUS_SUCCESS);

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        EXPECT_GE(output[i], 0u);
        sum += output[i];
    }
    double sample_mean = sum / n;
    ASSERT_NEAR(sample_mean, lambda, 0.2);

    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
    free(output);
}

TEST(cuRAND, GenerateUniformDoubleDevice) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 7890ULL), CURAND_STATUS_SUCCESS);

    const size_t n = 10;
    double* output;
    ASSERT_EQ(cudaMalloc(&output, n * sizeof(double)), cudaSuccess);

    ASSERT_EQ(curandGenerateUniformDouble(generator, output, n), CURAND_STATUS_SUCCESS);

    double host_output[n];
    ASSERT_EQ(cudaMemcpy(host_output, output, n * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_GE(host_output[i], 0.0);
        EXPECT_LT(host_output[i], 1.0);
    }

    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(cudaFree(output), cudaSuccess);
}

TEST(cuRAND, GenerateUniformDoubleHost) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 7890ULL), CURAND_STATUS_SUCCESS);

    const size_t n = 10;
    double* output = (double*)malloc(n * sizeof(double));
    ASSERT_NE(output, nullptr);

    ASSERT_EQ(curandGenerateUniformDouble(generator, output, n), CURAND_STATUS_SUCCESS);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_GE(output[i], 0.0);
        EXPECT_LT(output[i], 1.0);
    }

    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
    free(output);
}

TEST(cuRAND, GenerateNormalDoubleDevice) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 2468ULL), CURAND_STATUS_SUCCESS);

    const size_t n = 1000;
    double* output;
    ASSERT_EQ(cudaMalloc(&output, n * sizeof(double)), cudaSuccess);

    const double mean = 10.0;
    const double stddev = 3.0;
    ASSERT_EQ(curandGenerateNormalDouble(generator, output, n, mean, stddev), CURAND_STATUS_SUCCESS);

    double host_output[n];
    ASSERT_EQ(cudaMemcpy(host_output, output, n * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);

    // Basic sanity: check mean and stddev roughly
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += host_output[i];
    }
    double sample_mean = sum / n;
    ASSERT_NEAR(sample_mean, mean, 0.2);

    double variance_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = host_output[i] - mean;
        variance_sum += diff * diff;
    }
    double sample_stddev = sqrt(variance_sum / n);
    ASSERT_NEAR(sample_stddev, stddev, 0.3);

    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(cudaFree(output), cudaSuccess);
}

TEST(cuRAND, GenerateNormalDoubleHost) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 2468ULL), CURAND_STATUS_SUCCESS);

    const size_t n = 1000;
    double* output = (double*)malloc(n * sizeof(double));
    ASSERT_NE(output, nullptr);

    const double mean = 10.0;
    const double stddev = 3.0;
    ASSERT_EQ(curandGenerateNormalDouble(generator, output, n, mean, stddev), CURAND_STATUS_SUCCESS);

    // Basic sanity: check mean and stddev roughly
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += output[i];
    }
    double sample_mean = sum / n;
    ASSERT_NEAR(sample_mean, mean, 0.2);

    double variance_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = output[i] - mean;
        variance_sum += diff * diff;
    }
    double sample_stddev = sqrt(variance_sum / n);
    ASSERT_NEAR(sample_stddev, stddev, 0.3);

    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
    free(output);
}

TEST(cuRAND, GenerateLogNormalDoubleDevice) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 1357ULL), CURAND_STATUS_SUCCESS);

    const size_t n = 1000;
    double* output;
    ASSERT_EQ(cudaMalloc(&output, n * sizeof(double)), cudaSuccess);

    const double mean = 0.0;   // mean of underlying normal
    const double stddev = 0.5; // stddev of underlying normal
    ASSERT_EQ(curandGenerateLogNormalDouble(generator, output, n, mean, stddev), CURAND_STATUS_SUCCESS);

    double host_output[n];
    ASSERT_EQ(cudaMemcpy(host_output, output, n * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);

    // All outputs should be positive
    for (size_t i = 0; i < n; ++i) {
        EXPECT_GT(host_output[i], 0.0);
    }

    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(cudaFree(output), cudaSuccess);
}

TEST(cuRAND, GenerateLogNormalDoubleHost) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 1357ULL), CURAND_STATUS_SUCCESS);

    const size_t n = 1000;
    double* output = (double*)malloc(n * sizeof(double));
    ASSERT_NE(output, nullptr);

    const double mean = 0.0;
    const double stddev = 0.5;
    ASSERT_EQ(curandGenerateLogNormalDouble(generator, output, n, mean, stddev), CURAND_STATUS_SUCCESS);

    // All outputs should be positive
    for (size_t i = 0; i < n; ++i) {
        EXPECT_GT(output[i], 0.0);
    }

    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
    free(output);
}