FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt update && apt install -y --no-install-recommends build-essential clang-12 libxmu-dev libxi-dev libgl-dev libosmesa-dev git liblog4cplus-dev librdmacm-dev libibverbs-dev cmake pkg-config curl && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/tgasla/gvirtus.git
RUN mkdir gvirtus/build && cd gvirtus/build && cmake .. && make && make install
ENV GVIRTUS_HOME=/root/GVirtuS
ENV GVIRTUS_LOGLEVEL=0
ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${LD_LIBRARY_PATH}

# DISABLE IF USING DOCKER COMPOSE
COPY ./properties.json ${GVIRTUS_HOME}/etc/
#RUN apt update && apt install python3 -y
#COPY ./server.py .
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["${GVIRTUS_HOME}/bin/gvirtus-backend ${GVIRTUS_HOME}/etc/properties.json"]
#CMD ["python3 ./server.py"]
##########