FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt update && apt install -y --no-install-recommends build-essential libxmu-dev libxi-dev libgl-dev libosmesa-dev git liblog4cplus-dev librdmacm-dev libibverbs-dev autotools-dev automake cmake pkg-config curl && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/tgasla/gvirtus.git
ENV GVIRTUS_HOME=/usr/local/gvirtus
RUN mkdir gvirtus/build && cd gvirtus/build && cmake .. && make && make install
ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}
ENV GVIRTUS_LOGLEVEL=0
RUN sed -i 's/"server_address": "127.0.0.1"/"server_address": "0.0.0.0"/' ${GVIRTUS_HOME}/etc/properties.json