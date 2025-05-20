FROM taslanidis/gvirtus:cuda11.8.0-cudnn8-ubuntu22.04
# Install essential tools and dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    libgtest-dev \
    gdb \
    valgrind \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tests
COPY CMakeLists.txt .
COPY test_cudart.cu .
COPY test_cublas.cu .
COPY test_curand.cu .
COPY test_cufft.cu .
ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}
# Build the tests
RUN mkdir build && cd build && cmake .. && make
# Run the tests when container starts
CMD ["./build/test_cuda_libs"]
