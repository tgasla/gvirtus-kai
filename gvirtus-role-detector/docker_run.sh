#!/bin/bash

docker run --rm -it --user $(id -u):$(id -g) --net=host -e NODE_NAME=$NODE_NAME -e HOME=/home/$USER -v $HOME:/home/$USER taslanidis/gvirtus-node-setup
