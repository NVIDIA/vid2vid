#!/usr/bin/env python2.7

import caffe
from caffe.proto import caffe_pb2
import sys, os

import torch
import torch.nn as nn

import argparse, tempfile
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('caffe_model', help='input model in hdf5 or caffemodel format')
parser.add_argument('prototxt_template',help='prototxt template')
parser.add_argument('flownet2_pytorch', help='path to flownet2-pytorch')

args = parser.parse_args()

args.rgb_max = 255
args.fp16 = False
args.grads = {}

# load models
sys.path.append(args.flownet2_pytorch)

import models
from utils.param_utils import *

width = 256
height = 256
keys = {'TARGET_WIDTH': width, 
        'TARGET_HEIGHT': height,
        'ADAPTED_WIDTH':width,
        'ADAPTED_HEIGHT':height,
        'SCALE_WIDTH':1.,
        'SCALE_HEIGHT':1.,}

template = '\n'.join(np.loadtxt(args.prototxt_template, dtype=str, delimiter='\n'))
for k in keys:
    template = template.replace('$%s$'%(k),str(keys[k]))

prototxt = tempfile.NamedTemporaryFile(mode='w', delete=True)
prototxt.write(template)
prototxt.flush()

net = caffe.Net(prototxt.name, args.caffe_model, caffe.TEST)

weights = {}
biases = {}

for k, v in list(net.params.items()):
    weights[k] = np.array(v[0].data).reshape(v[0].data.shape)
    biases[k] = np.array(v[1].data).reshape(v[1].data.shape)
    print((k, weights[k].shape, biases[k].shape))

if 'FlowNet2/' in args.caffe_model:
    model = models.FlowNet2(args)

    parse_flownetc(model.flownetc.modules(), weights, biases)
    parse_flownets(model.flownets_1.modules(), weights, biases, param_prefix='net2_')
    parse_flownets(model.flownets_2.modules(), weights, biases, param_prefix='net3_')
    parse_flownetsd(model.flownets_d.modules(), weights, biases, param_prefix='netsd_')
    parse_flownetfusion(model.flownetfusion.modules(), weights, biases, param_prefix='fuse_')

    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(args.flownet2_pytorch, 'FlowNet2_checkpoint.pth.tar'))

elif 'FlowNet2-C/' in args.caffe_model:
    model = models.FlowNet2C(args)

    parse_flownetc(model.modules(), weights, biases)
    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(args.flownet2_pytorch, 'FlowNet2-C_checkpoint.pth.tar'))

elif 'FlowNet2-CS/' in args.caffe_model:
    model = models.FlowNet2CS(args)

    parse_flownetc(model.flownetc.modules(), weights, biases)
    parse_flownets(model.flownets_1.modules(), weights, biases, param_prefix='net2_')

    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(args.flownet2_pytorch, 'FlowNet2-CS_checkpoint.pth.tar'))

elif 'FlowNet2-CSS/' in args.caffe_model:
    model = models.FlowNet2CSS(args)

    parse_flownetc(model.flownetc.modules(), weights, biases)
    parse_flownets(model.flownets_1.modules(), weights, biases, param_prefix='net2_')
    parse_flownets(model.flownets_2.modules(), weights, biases, param_prefix='net3_')

    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(args.flownet2_pytorch, 'FlowNet2-CSS_checkpoint.pth.tar'))

elif 'FlowNet2-CSS-ft-sd/' in args.caffe_model:
    model = models.FlowNet2CSS(args)

    parse_flownetc(model.flownetc.modules(), weights, biases)
    parse_flownets(model.flownets_1.modules(), weights, biases, param_prefix='net2_')
    parse_flownets(model.flownets_2.modules(), weights, biases, param_prefix='net3_')

    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(args.flownet2_pytorch, 'FlowNet2-CSS-ft-sd_checkpoint.pth.tar'))

elif 'FlowNet2-S/' in args.caffe_model:
    model = models.FlowNet2S(args)

    parse_flownetsonly(model.modules(), weights, biases, param_prefix='')
    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(args.flownet2_pytorch, 'FlowNet2-S_checkpoint.pth.tar'))

elif 'FlowNet2-SD/' in args.caffe_model:
    model = models.FlowNet2SD(args)

    parse_flownetsd(model.modules(), weights, biases, param_prefix='')

    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(args.flownet2_pytorch, 'FlowNet2-SD_checkpoint.pth.tar'))

else:
    print(('model type cound not be determined from input caffe model %s'%(args.caffe_model)))
    quit()
print(("done converting ", args.caffe_model))