#!/usr/bin/env bash

#docker run --rm --name gvirtus-backend -p 34567:9999 --entrypoint /usr/local/gvirtus/bin/gvirtus-backend -it taslanidis/gvirtus:cuda11.8.0-cudnn8-ubuntu22.04 /usr/local/gvirtus/etc/properties.json
docker run --rm --name gvirtus-backend -p 34567:9999 -it taslanidis/gvirtus:cuda11.8.0-cudnn8-ubuntu22.04
