FROM taslanidis/gvirtus:cuda11.8.0-cudnn8-ubuntu22.04
WORKDIR /app
RUN useradd -m appuser
COPY cuda_examples .
RUN chown -R appuser:appuser ${GVIRTUS_HOME}
RUN chown -R appuser:appuser /app
ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}
RUN apt update && apt install -y gdb valgrind
USER appuser
RUN nvcc -o example example.cu -L ${GVIRTUS_HOME}/lib/frontend -lcublas -lcudnn -lcurand -lcufft --cudart=shared
RUN nvcc -o cudart_example cudart_example.cu -L ${GVIRTUS_HOME}/lib/frontend -lcublas -lcudnn -lcurand -lcufft --cudart=shared
RUN nvcc -o full_example full_example.cu -L ${GVIRTUS_HOME}/lib/frontend -lcublas -lcudnn -lcurand -lcufft --cudart=shared
RUN nvcc -o cudart_cublas_example cudart_cublas_example.cu -L ${GVIRTUS_HOME}/lib/frontend -lcublas -lcudnn -lcurand -lcufft --cudart=shared