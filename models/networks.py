### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import copy

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):        
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, prev_output_nc, ngf, which_model_netG, n_downsampling, norm, scale, gpu_ids=[], opt=[]):
    netG = None    
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'global':        
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsampling, opt.n_blocks, norm_layer)            
    elif which_model_netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsampling, opt.n_blocks, opt.n_local_enhancers, opt.n_blocks_local, norm_layer)
    elif which_model_netG == 'global_with_features':    
        netG = Global_with_z(input_nc, output_nc, opt.feat_num, ngf, n_downsampling, opt.n_blocks, norm_layer)     
    elif which_model_netG == 'local_with_features':    
        netG = Local_with_z(input_nc, output_nc, opt.feat_num, ngf, n_downsampling, opt.n_blocks, opt.n_local_enhancers, opt.n_blocks_local, norm_layer)

    elif which_model_netG == 'composite':
        netG = CompositeGenerator(opt, input_nc, output_nc, prev_output_nc, ngf, n_downsampling, opt.n_blocks, opt.fg, opt.no_flow, norm_layer)
    elif which_model_netG == 'compositeLocal':
        netG = CompositeLocalGenerator(opt, input_nc, output_nc, prev_output_nc, ngf, n_downsampling, opt.n_blocks_local, opt.fg, opt.no_flow, 
                                       norm_layer, scale=scale)    
    elif which_model_netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsampling, norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)

    #print_network(netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, num_D, getIntermFeat)   
    #print_network(netD)
    if len(gpu_ids) > 0:    
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def get_grid(batchsize, rows, cols, gpu_id=0, dtype=torch.float32):
    hor = torch.linspace(-1.0, 1.0, cols)
    hor.requires_grad = False
    hor = hor.view(1, 1, 1, cols)
    hor = hor.expand(batchsize, 1, rows, cols)
    ver = torch.linspace(-1.0, 1.0, rows)
    ver.requires_grad = False
    ver = ver.view(1, 1, rows, 1)
    ver = ver.expand(batchsize, 1, rows, cols)

    t_grid = torch.cat([hor, ver], 1)
    t_grid.requires_grad = False

    if dtype == torch.float16: t_grid = t_grid.half()
    return t_grid.cuda(gpu_id)

##############################################################################
# Classes
##############################################################################
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def grid_sample(self, input1, input2):
        if self.opt.fp16: # not sure if it's necessary
            return torch.nn.functional.grid_sample(input1.float(), input2.float(), mode='bilinear', padding_mode='border').half()
        else:
            return torch.nn.functional.grid_sample(input1, input2, mode='bilinear', padding_mode='border')

    def resample(self, image, flow):        
        b, c, h, w = image.size()        
        if not hasattr(self, 'grid') or self.grid.size() != flow.size():
            self.grid = get_grid(b, h, w, gpu_id=flow.get_device(), dtype=flow.dtype)            
        flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)        
        final_grid = (self.grid + flow).permute(0, 2, 3, 1).cuda(image.get_device())
        output = self.grid_sample(image, final_grid)
        return output

class CompositeGenerator(BaseNetwork):
    def __init__(self, opt, input_nc, output_nc, prev_output_nc, ngf, n_downsampling, n_blocks, use_fg_model=False, no_flow=False,
                norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(CompositeGenerator, self).__init__()                
        self.opt = opt
        self.n_downsampling = n_downsampling
        self.use_fg_model = use_fg_model
        self.no_flow = no_flow
        activation = nn.ReLU(True)
        
        if use_fg_model:
            ### individial image generation
            ngf_indv = ngf // 2 if n_downsampling > 2 else ngf
            indv_nc = input_nc
            indv_down = [nn.ReflectionPad2d(3), nn.Conv2d(indv_nc, ngf_indv, kernel_size=7, padding=0), 
                         norm_layer(ngf_indv), activation]        
            for i in range(n_downsampling):
                mult = 2**i
                indv_down += [nn.Conv2d(ngf_indv*mult, ngf_indv*mult*2, kernel_size=3, stride=2, padding=1), 
                              norm_layer(ngf_indv*mult*2), activation]

            indv_res = []
            mult = 2**n_downsampling
            for i in range(n_blocks):                
                indv_res += [ResnetBlock(ngf_indv * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            
            indv_up = []
            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)            
                indv_up += [nn.ConvTranspose2d(ngf_indv*mult, ngf_indv*mult//2, kernel_size=3, stride=2, padding=1, output_padding=1),
                            norm_layer(ngf_indv*mult//2), activation]                                
            indv_final = [nn.ReflectionPad2d(3), nn.Conv2d(ngf_indv, output_nc, kernel_size=7, padding=0), nn.Tanh()]        

        ### flow and image generation
        ### downsample        
        model_down_seg = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2**i
            model_down_seg += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                               norm_layer(ngf * mult * 2), activation]  

        mult = 2**n_downsampling
        for i in range(n_blocks - n_blocks//2):
            model_down_seg += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        model_down_img = [nn.ReflectionPad2d(3), nn.Conv2d(prev_output_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model_down_img += copy.deepcopy(model_down_seg[4:])
    
        ### resnet blocks
        model_res_img = []
        for i in range(n_blocks//2):
            model_res_img += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        if not no_flow:
            model_res_flow = copy.deepcopy(model_res_img)        

        ### upsample
        model_up_img = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_up_img += [nn.ConvTranspose2d(ngf*mult, ngf*mult//2, kernel_size=3, stride=2, padding=1, output_padding=1),
                             norm_layer(ngf*mult//2), activation]                    
        model_final_img = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        if not no_flow:
            model_up_flow = copy.deepcopy(model_up_img)
            model_final_flow = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 2, kernel_size=7, padding=0)]                
            model_final_w = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()] 

        if use_fg_model:
            self.indv_down = nn.Sequential(*indv_down)
            self.indv_res = nn.Sequential(*indv_res)
            self.indv_up = nn.Sequential(*indv_up)
            self.indv_final = nn.Sequential(*indv_final)

        self.model_down_seg = nn.Sequential(*model_down_seg)        
        self.model_down_img = nn.Sequential(*model_down_img)        
        self.model_res_img = nn.Sequential(*model_res_img)
        self.model_up_img = nn.Sequential(*model_up_img)
        self.model_final_img = nn.Sequential(*model_final_img)

        if not no_flow:
            self.model_res_flow = nn.Sequential(*model_res_flow)        
            self.model_up_flow = nn.Sequential(*model_up_flow)                
            self.model_final_flow = nn.Sequential(*model_final_flow)                       
            self.model_final_w = nn.Sequential(*model_final_w)

    def forward(self, input, img_prev, mask, img_feat_coarse, flow_feat_coarse, img_fg_feat_coarse, use_raw_only):
        downsample = self.model_down_seg(input) + self.model_down_img(img_prev)
        img_feat = self.model_up_img(self.model_res_img(downsample))
        img_raw = self.model_final_img(img_feat)

        flow = weight = flow_feat = None
        if not self.no_flow:
            res_flow = self.model_res_flow(downsample)                
            flow_feat = self.model_up_flow(res_flow)                                                              
            flow = self.model_final_flow(flow_feat) * 20
            weight = self.model_final_w(flow_feat)  

        gpu_id = img_feat.get_device()
        if use_raw_only or self.no_flow:
            img_final = img_raw
        else:
            img_warp = self.resample(img_prev[:,-3:,...].cuda(gpu_id), flow).cuda(gpu_id)        
            weight_ = weight.expand_as(img_raw)
            img_final = img_raw * weight_ + img_warp * (1-weight_)
        
        img_fg_feat = None
        if self.use_fg_model:
            img_fg_feat = self.indv_up(self.indv_res(self.indv_down(input)))
            img_fg = self.indv_final(img_fg_feat)

            mask = mask.cuda(gpu_id).expand_as(img_raw)            
            img_final = img_fg * mask + img_final * (1-mask) 
            img_raw = img_fg * mask + img_raw * (1-mask)                 

        return img_final, flow, weight, img_raw, img_feat, flow_feat, img_fg_feat

class CompositeLocalGenerator(BaseNetwork):
    def __init__(self, opt, input_nc, output_nc, prev_output_nc, ngf, n_downsampling, n_blocks_local, use_fg_model=False, no_flow=False,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect', scale=1):        
        super(CompositeLocalGenerator, self).__init__()                
        self.opt = opt
        self.use_fg_model = use_fg_model
        self.no_flow = no_flow
        self.scale = scale    
        activation = nn.ReLU(True)
        
        if use_fg_model:
            ### individial image generation        
            ngf_indv = ngf // 2 if n_downsampling > 2 else ngf
            indv_down = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_indv, kernel_size=7, padding=0), norm_layer(ngf_indv), activation,
                         nn.Conv2d(ngf_indv, ngf_indv*2, kernel_size=3, stride=2, padding=1), norm_layer(ngf_indv*2), activation]        

            indv_up = []
            for i in range(n_blocks_local):
                indv_up += [ResnetBlock(ngf_indv*2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
                    
            indv_up += [nn.ConvTranspose2d(ngf_indv*2, ngf_indv, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf_indv), activation]                            
            indv_final = [nn.ReflectionPad2d(3), nn.Conv2d(ngf_indv, output_nc, kernel_size=7, padding=0), nn.Tanh()]        


        ### flow and image generation
        ### downsample
        model_down_seg = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation,
                          nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1), norm_layer(ngf*2), activation]                  
        model_down_img = [nn.ReflectionPad2d(3), nn.Conv2d(prev_output_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation,
                          nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1), norm_layer(ngf*2), activation]        

        ### resnet blocks
        model_up_img = []        
        for i in range(n_blocks_local):
            model_up_img += [ResnetBlock(ngf*2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]            

        ### upsample        
        up = [nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf), activation]        
        model_up_img += up
        model_final_img = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        if not no_flow:
            model_up_flow = copy.deepcopy(model_up_img)        
            model_final_flow = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 2, kernel_size=7, padding=0)]        
            model_final_w = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()] 

        if use_fg_model:
            self.indv_down = nn.Sequential(*indv_down)        
            self.indv_up = nn.Sequential(*indv_up)
            self.indv_final = nn.Sequential(*indv_final)

        self.model_down_seg = nn.Sequential(*model_down_seg)        
        self.model_down_img = nn.Sequential(*model_down_img)        
        self.model_up_img = nn.Sequential(*model_up_img)
        self.model_final_img = nn.Sequential(*model_final_img)

        if not no_flow:
            self.model_up_flow = nn.Sequential(*model_up_flow)                
            self.model_final_flow = nn.Sequential(*model_final_flow)                     
            self.model_final_w = nn.Sequential(*model_final_w)        

    def forward(self, input, img_prev, mask, img_feat_coarse, flow_feat_coarse, img_fg_feat_coarse, use_raw_only):
        flow_multiplier = 20 * (2 ** self.scale)        
        down_img = self.model_down_seg(input) + self.model_down_img(img_prev)
        img_feat = self.model_up_img(down_img + img_feat_coarse)        
        img_raw = self.model_final_img(img_feat)

        flow = weight = flow_feat = None
        if not self.no_flow:
            down_flow = down_img
            flow_feat = self.model_up_flow(down_flow + flow_feat_coarse)            
            flow = self.model_final_flow(flow_feat) * flow_multiplier
            weight = self.model_final_w(flow_feat)

        gpu_id = img_feat.get_device()
        if use_raw_only or self.no_flow:
            img_final = img_raw
        else:                                    
            img_warp = self.resample(img_prev[:,-3:,...].cuda(gpu_id), flow).cuda(gpu_id)
            weight_ = weight.expand_as(img_raw)
            img_final = img_raw * weight_ + img_warp * (1-weight_)

        img_fg_feat = None
        if self.use_fg_model:
            img_fg_feat = self.indv_up(self.indv_down(input) + img_fg_feat_coarse)        
            img_fg = self.indv_final(img_fg_feat)
            mask = mask.cuda(gpu_id).expand_as(img_raw)
            img_final = img_fg * mask + img_final * (1-mask)
            img_raw = img_fg * mask + img_raw * (1-mask)         

        return img_final, flow, weight, img_raw, img_feat, flow_feat, img_fg_feat

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)
        ch_max = 1024        
        
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(min(ch_max, ngf * mult), min(ch_max, ngf * mult * 2), kernel_size=3, stride=2, padding=1),
                       norm_layer(min(ch_max, ngf * mult * 2)), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(min(ch_max, ngf * mult), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(min(ch_max, ngf * mult), min(ch_max, int(ngf * mult / 2)), 
                         kernel_size=3, stride=2, padding=1, output_padding=1),
                         norm_layer(min(ch_max, int(ngf * mult / 2))), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)        

    def forward(self, input, feat=None):
        if feat is not None:
            input = torch.cat([input, feat], dim=1)
        output = self.model(input)                
        return output

class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers        
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_final = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
                model_upsample += model_final
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))            

            ngf_global = ngf * (2**(n_local_enhancers-n)) * 2            
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input, feat_map=None):
        if feat_map is not None:
            input = torch.cat([input, feat_map], dim=1)

        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)        
        return output_prev

class Global_with_z(nn.Module):
    def __init__(self, input_nc, output_nc, nz, ngf=64, n_downsample_G=3, n_blocks=9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(Global_with_z, self).__init__()                
        self.n_downsample_G = n_downsample_G        
        max_ngf = 1024
        activation = nn.ReLU(True)

        # downsample model
        model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc + nz, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsample_G):
            mult = 2 ** i
            model_downsample += [nn.Conv2d(min(ngf * mult, max_ngf), min(ngf * mult * 2, max_ngf), kernel_size=3, stride=2, padding=1),
                                 norm_layer(min(ngf * mult * 2, max_ngf)), activation]

        # internal model
        model_resnet = []
        mult = 2 ** n_downsample_G
        for i in range(n_blocks):
            model_resnet += [ResnetBlock(min(ngf*mult, max_ngf) + nz, padding_type=padding_type, norm_layer=norm_layer)]

        # upsample model        
        model_upsample = []
        for i in range(n_downsample_G):
            mult = 2 ** (n_downsample_G - i)
            input_ngf = min(ngf * mult, max_ngf)
            if i == 0:
                input_ngf += nz * 2
            model_upsample += [nn.ConvTranspose2d(input_ngf, min((ngf * mult // 2), max_ngf), kernel_size=3, stride=2, 
                               padding=1, output_padding=1), norm_layer(min((ngf * mult // 2), max_ngf)), activation]        

        model_upsample_conv = [nn.ReflectionPad2d(3), nn.Conv2d(ngf + nz, output_nc, kernel_size=7), nn.Tanh()]

        self.model_downsample = nn.Sequential(*model_downsample)
        self.model_resnet = nn.Sequential(*model_resnet)        
        self.model_upsample = nn.Sequential(*model_upsample)
        self.model_upsample_conv = nn.Sequential(*model_upsample_conv)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)        

    def forward(self, x, z):
        z_downsample = z
        for i in range(self.n_downsample_G):
            z_downsample = self.downsample(z_downsample)
        downsample = self.model_downsample(torch.cat([x, z], dim=1))                
        resnet = self.model_resnet(torch.cat([downsample, z_downsample], dim=1))                
        upsample = self.model_upsample(torch.cat([resnet, z_downsample], dim=1))
        return self.model_upsample_conv(torch.cat([upsample, z], dim=1))

class Local_with_z(nn.Module):
    def __init__(self, input_nc, output_nc, nz, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(Local_with_z, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        self.n_downsample_global = n_downsample_global
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = Global_with_z(input_nc, output_nc, nz, ngf_global, n_downsample_global, n_blocks_global, norm_layer)        
        self.model_downsample = model_global.model_downsample
        self.model_resnet = model_global.model_resnet
        self.model_upsample = model_global.model_upsample

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            if n == n_local_enhancers:
                input_nc += nz
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            input_ngf = ngf_global * 2
            if n == 1:            
                input_ngf += nz
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(input_ngf, padding_type=padding_type, norm_layer=norm_layer)]
            ### upsample            
            model_upsample += [nn.ConvTranspose2d(input_ngf, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]              
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        ### final convolution        
        model_final = [nn.ReflectionPad2d(3), nn.Conv2d(ngf + nz, output_nc, kernel_size=7), nn.Tanh()]
        self.model_final = nn.Sequential(*model_final)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input, z): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### create downsampled z
        z_downsampled_local = z
        for i in range(self.n_local_enhancers):
            z_downsampled_local = self.downsample(z_downsampled_local)
        z_downsampled_global = z_downsampled_local
        for i in range(self.n_downsample_global):
            z_downsampled_global = self.downsample(z_downsampled_global)

        ### output at coarest level
        x = input_downsampled[-1]
        global_downsample = self.model_downsample(torch.cat([x, z_downsampled_local], dim=1))                
        global_resnet = self.model_resnet(torch.cat([global_downsample, z_downsampled_global], dim=1))                
        global_upsample = self.model_upsample(torch.cat([global_resnet, z_downsampled_global], dim=1))

        ### build up one layer at a time
        output_prev = global_upsample
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            # fetch models
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            # get input image
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]
            if n_local_enhancers == self.n_local_enhancers:
                input_i = torch.cat([input_i, z], dim=1)            
            # combine features from different resolutions
            combined_input = model_downsample(input_i) + output_prev
            if n_local_enhancers == 1:
                combined_input = torch.cat([combined_input, z_downsampled_local], dim=1)
            # upsample features
            output_prev = model_upsample(combined_input)

        # final convolution
        output = self.model_final(torch.cat([output_prev, z], dim=1))
        return output 

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()        
        for b in range(input.size()[0]):            
            inst_list = np.unique(inst[b].cpu().numpy().astype(int))            
            for i in inst_list:            
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4                
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]                    
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)                    
                    ### add random noise to output feature
                    #mean_feat += torch.normal(torch.zeros_like(mean_feat), 0.05 * torch.ones_like(mean_feat)).cuda()                    
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat
        return outputs_mean

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        ndf_max = 64
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, min(ndf_max, ndf*(2**(num_D-1-i))), n_layers, norm_layer,
                                       getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)        

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]            
            for i in range(len(model)):
                result.append(model[i](result[-1]))            
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))                                
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)                    
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)            

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        


##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor        
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None        
        gpu_id = input.get_device()
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)                
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda(gpu_id)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y):
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self, label_nc):
        super(CrossEntropyLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss2d()

    def forward(self, output, label):
        label = label.long().max(1)[1]        
        output = self.softmax(output)
        return self.criterion(output, label)

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target, mask):        
        mask = mask.expand(-1, input.size()[1], -1, -1)
        loss = self.criterion(input * mask, target * mask)
        return loss

class MultiscaleL1Loss(nn.Module):
    def __init__(self, scale=5):
        super(MultiscaleL1Loss, self).__init__()
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        #self.weights = [0.5, 1, 2, 8, 32]
        self.weights = [1, 0.5, 0.25, 0.125, 0.125]
        self.weights = self.weights[:scale]

    def forward(self, input, target, mask=None):
        loss = 0
        if mask is not None:
            mask = mask.expand(-1, input.size()[1], -1, -1)
        for i in range(len(self.weights)):
            if mask is not None:                
                loss += self.weights[i] * self.criterion(input * mask, target * mask)
            else:
                loss += self.weights[i] * self.criterion(input, target)
            if i != len(self.weights)-1:
                input = self.downsample(input)
                target = self.downsample(target)
                if mask is not None:
                    mask = self.downsample(mask)
        return loss

from torchvision import models
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out