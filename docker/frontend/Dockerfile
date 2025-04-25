FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt update && apt install -y --no-install-recommends build-essential libxmu-dev libxi-dev libgl-dev libosmesa-dev git liblog4cplus-dev librdmacm-dev libibverbs-dev autotools-dev automake cmake pkg-config curl && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/tgasla/gvirtus.git
RUN mkdir gvirtus/build && cd gvirtus/build && cmake .. && make && make install
ENV GVIRTUS_HOME=/root/GVirtuS
ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}
ENV GVIRTUS_LOGLEVEL=0
ENV EXTRA_NVCCFLAGS="--cudart=shared"
ENV CUDA_VISIBLE_DEVICES=-1
COPY ./properties.json ${GVIRTUS_HOME}/etc/
COPY full_example.cu .
RUN nvcc full_example.cu -L ${GVIRTUS_HOME}/lib -L ${GVIRTUS_HOME}/lib/frontend -lcublas -lcufft -lcudnn -lcurand -o full_example --cudart=shared
RUN chmod +x full_example
ENTRYPOINT ["/bin/bash" , "-c"]
CMD ["LD_PRELOAD=$(echo ${GVIRTUS_HOME}/lib/frontend/*.so | tr ' ' ':') ./full_example"]
