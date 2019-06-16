#!/bin/bash

FN2PYTORCH=${1:-/}

# install custom layers
sudo nvidia-docker build -t $USER/pytorch:CUDA8-py27 .
sudo nvidia-docker run --rm -ti --volume=${FN2PYTORCH}:/flownet2-pytorch:rw --workdir=/flownet2-pytorch $USER/pytorch:CUDA8-py27 /bin/bash -c "./install.sh"

# convert FlowNet2-C, CS, CSS, CSS-ft-sd, SD, S and 2 to PyTorch
sudo nvidia-docker run -ti --volume=${FN2PYTORCH}:/fn2pytorch:rw flownet2:latest /bin/bash -c "source /flownet2/flownet2/set-env.sh && cd /flownet2/flownet2/models && \
python /fn2pytorch/convert.py ./FlowNet2-C/FlowNet2-C_weights.caffemodel ./FlowNet2-C/FlowNet2-C_deploy.prototxt.template  /fn2pytorch && 
python /fn2pytorch/convert.py ./FlowNet2-CS/FlowNet2-CS_weights.caffemodel ./FlowNet2-CS/FlowNet2-CS_deploy.prototxt.template /fn2pytorch && \
python /fn2pytorch/convert.py ./FlowNet2-CSS/FlowNet2-CSS_weights.caffemodel.h5 ./FlowNet2-CSS/FlowNet2-CSS_deploy.prototxt.template /fn2pytorch && \
python /fn2pytorch/convert.py ./FlowNet2-CSS-ft-sd/FlowNet2-CSS-ft-sd_weights.caffemodel.h5 ./FlowNet2-CSS-ft-sd/FlowNet2-CSS-ft-sd_deploy.prototxt.template /fn2pytorch && \
python /fn2pytorch/convert.py ./FlowNet2-SD/FlowNet2-SD_weights.caffemodel.h5 ./FlowNet2-SD/FlowNet2-SD_deploy.prototxt.template /fn2pytorch && \
python /fn2pytorch/convert.py ./FlowNet2-S/FlowNet2-S_weights.caffemodel.h5 ./FlowNet2-S/FlowNet2-S_deploy.prototxt.template /fn2pytorch && \
python /fn2pytorch/convert.py ./FlowNet2/FlowNet2_weights.caffemodel.h5 ./FlowNet2/FlowNet2_deploy.prototxt.template /fn2pytorch"