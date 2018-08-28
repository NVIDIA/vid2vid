#!/bin/bash
sudo nvidia-docker build -t vid2vid:CUDA8-py35 .

#run first time without /bin/bash on the end to do install from CMD in Dockerfile
sudo nvidia-docker run --rm -ti --ipc=host -v $(pwd):/vid2vid --workdir=/vid2vid vid2vid:CUDA8-py35
