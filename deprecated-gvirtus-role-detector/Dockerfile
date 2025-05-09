FROM bitnami/kubectl:latest

ARG UID
ARG GID
ARG USER

USER root

RUN echo $UID
RUN echo $GID
RUN echo $USER
RUN apt-get update && apt-get install sudo && \
    groupadd -g ${GID} ${USER} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USER}

WORKDIR /app
COPY detect-gvirtus.sh .
ENTRYPOINT ["/app/detect-gvirtus.sh"]
