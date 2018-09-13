#!/bin/bash
sudo nvidia-docker build -t vid2vid:CUDA8-py35 .

sudo nvidia-docker run --rm -ti --ipc=host --shm-size 8G -v $(pwd):/vid2vid --workdir=/vid2vid vid2vid:CUDA8-py35 /bin/bash
