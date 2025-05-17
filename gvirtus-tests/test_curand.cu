#include <gtest/gtest.h>
#include <iostream>
#include <curand.h>

TEST(CurandTest, CreateGenerator) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
}

TEST(CurandTest, CreateGeneratorHost) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
}

TEST(CurandTest, SetSeed) {
    curandGenerator_t generator;
    ASSERT_EQ(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL), CURAND_STATUS_SUCCESS);
    ASSERT_EQ(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);
}

TEST(CurandTest, GenerateUniform) {
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}