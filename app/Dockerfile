FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
WORKDIR /app
RUN useradd -m appuser
COPY cudart_example.cu .
RUN chown -R appuser:appuser /app
USER appuser
RUN nvcc -o cudart_example cudart_example.cu -lcublas -lcudnn -lcurand -lcufft --cudart=shared
ENTRYPOINT ["./cudart_example"]
