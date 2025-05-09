#!/usr/bin/env bash
docker buildx build --platform linux/arm64,linux/amd64 --push -t taslanidis/gvirtus-node-setup --build-arg USER=$(id -un) --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
