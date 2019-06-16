import torch
import torch.nn as nn
import numpy as np

def parse_flownetc(modules, weights, biases):
    keys = [
    'conv1',
    'conv2',
    'conv3',
    'conv_redir',
    'conv3_1',
    'conv4',
    'conv4_1',
    'conv5',
    'conv5_1',
    'conv6',
    'conv6_1',
    
    'deconv5',
    'deconv4',
    'deconv3',
    'deconv2',
    
    'Convolution1',
    'Convolution2',
    'Convolution3',
    'Convolution4',
    'Convolution5',

    'upsample_flow6to5',
    'upsample_flow5to4',
    'upsample_flow4to3',
    'upsample_flow3to2',
    
    ]
    i = 0
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            weight = weights[keys[i]].copy()
            bias = biases[keys[i]].copy()
            if keys[i] == 'conv1':
                m.weight.data[:,:,:,:] = torch.from_numpy(np.flip(weight, axis=1).copy())
                m.bias.data[:] = torch.from_numpy(bias)
            else:
                m.weight.data[:,:,:,:] = torch.from_numpy(weight)
                m.bias.data[:] = torch.from_numpy(bias)                    

            i = i + 1
    return

def parse_flownets(modules, weights, biases, param_prefix='net2_'):
    keys = [
    'conv1',
    'conv2',
    'conv3',
    'conv3_1',
    'conv4',
    'conv4_1',
    'conv5',
    'conv5_1',
    'conv6',
    'conv6_1',
    
    'deconv5',
    'deconv4',
    'deconv3',
    'deconv2',
    
    'predict_conv6',
    'predict_conv5',
    'predict_conv4',
    'predict_conv3',
    'predict_conv2',

    'upsample_flow6to5',
    'upsample_flow5to4',
    'upsample_flow4to3',
    'upsample_flow3to2',
    ]
    for i, k in enumerate(keys):
        if 'upsample' in k:
            keys[i] = param_prefix + param_prefix + k
        else:
            keys[i] = param_prefix + k
    i = 0
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            weight = weights[keys[i]].copy()
            bias = biases[keys[i]].copy()
            if keys[i] == param_prefix+'conv1':
                m.weight.data[:,0:3,:,:] = torch.from_numpy(np.flip(weight[:,0:3,:,:], axis=1).copy())
                m.weight.data[:,3:6,:,:] = torch.from_numpy(np.flip(weight[:,3:6,:,:], axis=1).copy())
                m.weight.data[:,6:9,:,:] = torch.from_numpy(np.flip(weight[:,6:9,:,:], axis=1).copy())
                m.weight.data[:,9::,:,:] = torch.from_numpy(weight[:,9:,:,:].copy())
                if m.bias is not None:
                    m.bias.data[:] = torch.from_numpy(bias)
            else:
                m.weight.data[:,:,:,:] = torch.from_numpy(weight)
                if m.bias is not None:
                    m.bias.data[:] = torch.from_numpy(bias)
            i = i + 1
    return

def parse_flownetsonly(modules, weights, biases, param_prefix=''):
    keys = [
    'conv1',
    'conv2',
    'conv3',
    'conv3_1',
    'conv4',
    'conv4_1',
    'conv5',
    'conv5_1',
    'conv6',
    'conv6_1',
    
    'deconv5',
    'deconv4',
    'deconv3',
    'deconv2',
    
    'Convolution1',
    'Convolution2',
    'Convolution3',
    'Convolution4',
    'Convolution5',

    'upsample_flow6to5',
    'upsample_flow5to4',
    'upsample_flow4to3',
    'upsample_flow3to2',
    ]
    for i, k in enumerate(keys):
        if 'upsample' in k:
            keys[i] = param_prefix + param_prefix + k
        else:
            keys[i] = param_prefix + k
    i = 0
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            weight = weights[keys[i]].copy()
            bias = biases[keys[i]].copy()
            if keys[i] == param_prefix+'conv1':
                # print ("%s :"%(keys[i]), m.weight.size(), m.bias.size(), tf_w[keys[i]].shape[::-1])
                m.weight.data[:,0:3,:,:] = torch.from_numpy(np.flip(weight[:,0:3,:,:], axis=1).copy())
                m.weight.data[:,3:6,:,:] = torch.from_numpy(np.flip(weight[:,3:6,:,:], axis=1).copy())
                if m.bias is not None:
                    m.bias.data[:] = torch.from_numpy(bias)
            else:
                m.weight.data[:,:,:,:] = torch.from_numpy(weight)
                if m.bias is not None:
                    m.bias.data[:] = torch.from_numpy(bias)
            i = i + 1
    return

def parse_flownetsd(modules, weights, biases, param_prefix='netsd_'):
    keys = [
    'conv0',
    'conv1',
    'conv1_1',
    'conv2',
    'conv2_1',
    'conv3',
    'conv3_1',
    'conv4',
    'conv4_1',
    'conv5',
    'conv5_1',
    'conv6',
    'conv6_1',
    
    'deconv5',
    'deconv4',
    'deconv3',
    'deconv2',

    'interconv5',
    'interconv4',
    'interconv3',
    'interconv2',
    
    'Convolution1',
    'Convolution2',
    'Convolution3',
    'Convolution4',
    'Convolution5',

    'upsample_flow6to5',
    'upsample_flow5to4',
    'upsample_flow4to3',
    'upsample_flow3to2',
    ]
    for i, k in enumerate(keys):
        keys[i] = param_prefix + k

    i = 0
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            weight = weights[keys[i]].copy()
            bias = biases[keys[i]].copy()
            if keys[i] == param_prefix+'conv0':
                m.weight.data[:,0:3,:,:] = torch.from_numpy(np.flip(weight[:,0:3,:,:], axis=1).copy())
                m.weight.data[:,3:6,:,:] = torch.from_numpy(np.flip(weight[:,3:6,:,:], axis=1).copy())
                if m.bias is not None:
                    m.bias.data[:] = torch.from_numpy(bias)
            else:
                m.weight.data[:,:,:,:] = torch.from_numpy(weight)
                if m.bias is not None:
                    m.bias.data[:] = torch.from_numpy(bias)
            i = i + 1

    return

def parse_flownetfusion(modules, weights, biases, param_prefix='fuse_'):
    keys = [
    'conv0',
    'conv1',
    'conv1_1',
    'conv2',
    'conv2_1',

    'deconv1',
    'deconv0',

    'interconv1',
    'interconv0',
    
    '_Convolution5',
    '_Convolution6',
    '_Convolution7',

    'upsample_flow2to1',
    'upsample_flow1to0',
    ]
    for i, k in enumerate(keys):
        keys[i] = param_prefix + k

    i = 0
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            weight = weights[keys[i]].copy()
            bias = biases[keys[i]].copy()
            if keys[i] == param_prefix+'conv0':
                m.weight.data[:,0:3,:,:] = torch.from_numpy(np.flip(weight[:,0:3,:,:], axis=1).copy())
                m.weight.data[:,3::,:,:] = torch.from_numpy(weight[:,3:,:,:].copy())
                if m.bias is not None:
                    m.bias.data[:] = torch.from_numpy(bias)
            else:
                m.weight.data[:,:,:,:] = torch.from_numpy(weight)
                if m.bias is not None:
                    m.bias.data[:] = torch.from_numpy(bias)
            i = i + 1

    return
