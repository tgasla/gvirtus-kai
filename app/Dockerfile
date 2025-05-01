FROM taslanidis/gvirtus:11.8.0-cudnn8-devel-ubuntu22.04
WORKDIR /app
RUN useradd -m appuser
COPY cuda-examples/cudart_example.cu .
RUN chown -R appuser:appuser /app
USER appuser
RUN nvcc -o cudart_example cudart_example.cu -L ${GVIRTUS_HOME}/lib/frontend -lcublas -lcudnn -lcurand -lcufft --cudart=shared
ENTRYPOINT ["./cudart_example"]
