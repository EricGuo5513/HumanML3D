#!/bin/bash

sudo docker build -t humanml3d -f docker/Dockerfile .
sudo docker run --gpus all --ipc host -v $(pwd):/humanml3d -v /mnt/data:/data -it humanml3d bash
