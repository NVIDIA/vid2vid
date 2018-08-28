FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN apt-get update && apt-get install -y rsync htop git openssh-server python3-pip

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install --upgrade pip

#Torch and dependencies:
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp35-cp35m-linux_x86_64.whl 
RUN pip install torchvision cffi tensorboardX
RUN pip install tqdm scipy scikit-image colorama==0.3.7 
RUN pip install setproctitle pytz ipython

#vid2vid dependencies
RUN apt-get install libglib2.0-0 libsm6 libxrender1 -y
RUN pip install dominate requests opencv-python 

#first-time install
CMD python scripts/download_flownet2.py && python scripts/download_models_g1.py && bash scripts/test_1024_g1.sh && /bin/bash

