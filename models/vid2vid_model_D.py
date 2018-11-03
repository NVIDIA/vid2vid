### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
import sys
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
from .flownet2_pytorch.networks.resample2d_package.resample2d import Resample2d

class Vid2VidModelD(BaseModel):
    def name(self):
        return 'Vid2VidModelD'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)        
        gpu_split_id = opt.n_gpus_gen
        if opt.batchSize == 1:
            gpu_split_id += 1
        self.gpu_ids = (opt.gpu_ids[gpu_split_id:] + [opt.gpu_ids[0]]) if opt.n_gpus_gen != len(opt.gpu_ids) else opt.gpu_ids
        if not opt.debug:
            torch.backends.cudnn.benchmark = True    
        self.tD = opt.n_frames_D  
        self.output_nc = opt.output_nc        

        # define networks        
        # single image discriminator
        self.input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        if opt.use_instance:
            self.input_nc += 1
        netD_input_nc = self.input_nc + opt.output_nc
               
        self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm,
                                      opt.num_D, not opt.no_ganFeat, gpu_ids=self.gpu_ids)

        if opt.add_face_disc:            
            self.netD_f = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm,
                                            max(1, opt.num_D - 2), not opt.no_ganFeat, gpu_ids=self.gpu_ids)
                    
        # temporal discriminator
        netD_input_nc = opt.output_nc * opt.n_frames_D + 2 * (opt.n_frames_D-1)        
        for s in range(opt.n_scales_temporal):
            setattr(self, 'netD_T'+str(s), networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm,
                    opt.num_D, not opt.no_ganFeat, gpu_ids=self.gpu_ids))

        # flownet 2        
        self.resample = Resample2d()           

        print('---------- Networks initialized -------------') 
        print('-----------------------------------------------')

        # load networks
        if opt.continue_train or opt.load_pretrain:          
            self.load_network(self.netD, 'D', opt.which_epoch, opt.load_pretrain)            
            for s in range(opt.n_scales_temporal):
                self.load_network(getattr(self, 'netD_T'+str(s)), 'D_T'+str(s), opt.which_epoch, opt.load_pretrain)
            if opt.add_face_disc:
                self.load_network(self.netD_f, 'D_f', opt.which_epoch, opt.load_pretrain)
           
        # set loss functions and optimizers          
        self.old_lr = opt.lr
        # define loss functions
        self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.Tensor)   
        self.criterionFlow = networks.MaskedL1Loss()
        self.criterionWarp = networks.MaskedL1Loss()
        self.criterionFeat = torch.nn.L1Loss()
        if not opt.no_vgg:
            self.criterionVGG = networks.VGGLoss(self.gpu_ids[0])

        self.loss_names = ['G_VGG', 'G_GAN', 'G_GAN_Feat',                            
                           'D_real', 'D_fake',
                           'G_Warp', 'F_Flow', 'F_Warp', 'W']                
        self.loss_names_T = ['G_T_GAN', 'G_T_GAN_Feat', 'D_T_real', 'D_T_fake', 'G_T_Warp']     
        if opt.add_face_disc:
            self.loss_names += ['G_f_GAN', 'G_f_GAN_Feat', 'D_f_real', 'D_f_fake']

        # initialize optimizers D and D_T                                            
        params = list(self.netD.parameters())
        if opt.add_face_disc:
            params += list(self.netD_f.parameters())
        if opt.TTUR:                
            beta1, beta2 = 0, 0.9
            lr = opt.lr * 2
        else:
            beta1, beta2 = opt.beta1, 0.999
            lr = opt.lr
        self.optimizer_D = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))        

        for s in range(opt.n_scales_temporal):            
            params = list(getattr(self, 'netD_T'+str(s)).parameters())          
            optimizer_D_T = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))            
            setattr(self, 'optimizer_D_T'+str(s), optimizer_D_T)

    def compute_loss_D(self, netD, real_A, real_B, fake_B):        
        real_AB = torch.cat((real_A, real_B), dim=1)
        fake_AB = torch.cat((real_A, fake_B), dim=1)
        pred_real = netD.forward(real_AB)
        pred_fake = netD.forward(fake_AB.detach())
        loss_D_real = self.criterionGAN(pred_real, True) 
        loss_D_fake = self.criterionGAN(pred_fake, False)

        pred_fake = netD.forward(fake_AB)                       
        loss_G_GAN, loss_G_GAN_Feat = self.GAN_and_FM_loss(pred_real, pred_fake)      

        return loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat      

    def compute_loss_D_T(self, real_B, fake_B, flow_ref, conf_ref, scale_T):         
        netD_T = getattr(self, 'netD_T'+str(scale_T))
        real_B = real_B.view(-1, self.output_nc * self.tD, self.height, self.width)
        fake_B = fake_B.view(-1, self.output_nc * self.tD, self.height, self.width)        
        if flow_ref is not None:
            flow_ref = flow_ref.view(-1, 2 * (self.tD-1), self.height, self.width)                        
            real_B = torch.cat([real_B, flow_ref], dim=1)
            fake_B = torch.cat([fake_B, flow_ref], dim=1)
        pred_real = netD_T.forward(real_B)
        pred_fake = netD_T.forward(fake_B.detach())
        loss_D_T_real = self.criterionGAN(pred_real, True)            
        loss_D_T_fake = self.criterionGAN(pred_fake, False)        

        pred_fake = netD_T.forward(fake_B)                               
        loss_G_T_GAN, loss_G_T_GAN_Feat = self.GAN_and_FM_loss(pred_real, pred_fake)

        return loss_D_T_real, loss_D_T_fake, loss_G_T_GAN, loss_G_T_GAN_Feat

    def GAN_and_FM_loss(self, pred_real, pred_fake):
        ### GAN loss            
        loss_G_GAN = self.criterionGAN(pred_fake, True)                             

        # GAN feature matching loss
        loss_G_GAN_Feat = torch.zeros_like(loss_G_GAN)
        if not self.opt.no_ganFeat:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(min(len(pred_fake), self.opt.num_D)):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        return loss_G_GAN, loss_G_GAN_Feat

    def forward(self, scale_T, tensors_list):
        lambda_feat = self.opt.lambda_feat
        lambda_F = self.opt.lambda_F
        lambda_T = self.opt.lambda_T
        scale_S = self.opt.n_scales_spatial
        tD = self.opt.n_frames_D
        
        if scale_T > 0:
            real_B, fake_B, flow_ref, conf_ref = tensors_list
            _, _, _, self.height, self.width = real_B.size()
            loss_D_T_real, loss_D_T_fake, loss_G_T_GAN, loss_G_T_GAN_Feat = self.compute_loss_D_T(real_B, fake_B, 
                flow_ref/20, conf_ref, scale_T-1)            
            loss_G_T_Warp = torch.zeros_like(loss_G_T_GAN)

            loss_list = [loss_G_T_GAN, loss_G_T_GAN_Feat, loss_D_T_real, loss_D_T_fake, loss_G_T_Warp]
            loss_list = [loss.unsqueeze(0) for loss in loss_list]
            return loss_list            


        real_B, fake_B, fake_B_raw, real_A, real_B_prev, fake_B_prev, flow, weight, flow_ref, conf_ref = tensors_list
        _, _, self.height, self.width = real_B.size()

        ################### Flow loss #################
        if flow is not None:
            # similar to flownet flow        
            loss_F_Flow = self.criterionFlow(flow, flow_ref, conf_ref) * lambda_F / (2 ** (scale_S-1))        
            # warped prev image should be close to current image            
            real_B_warp = self.resample(real_B_prev, flow)                
            loss_F_Warp = self.criterionFlow(real_B_warp, real_B, conf_ref) * lambda_T
            
            ################## weight loss ##################
            loss_W = torch.zeros_like(weight)
            if self.opt.no_first_img:
                dummy0 = torch.zeros_like(weight)
                loss_W = self.criterionFlow(weight, dummy0, conf_ref)
        else:
            loss_F_Flow = loss_F_Warp = loss_W = torch.zeros_like(conf_ref)

        #################### fake_B loss ####################        
        ### VGG + GAN loss 
        loss_G_VGG = (self.criterionVGG(fake_B, real_B) * lambda_feat) if not self.opt.no_vgg else torch.zeros_like(loss_W)
        loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat = self.compute_loss_D(self.netD, real_A, real_B, fake_B)
        ### Warp loss
        fake_B_warp_ref = self.resample(fake_B_prev, flow_ref)
        loss_G_Warp = self.criterionWarp(fake_B, fake_B_warp_ref.detach(), conf_ref) * lambda_T
        
        if fake_B_raw is not None:
            if not self.opt.no_vgg:
                loss_G_VGG += self.criterionVGG(fake_B_raw, real_B) * lambda_feat        
            l_D_real, l_D_fake, l_G_GAN, l_G_GAN_Feat = self.compute_loss_D(self.netD, real_A, real_B, fake_B_raw)        
            loss_G_GAN += l_G_GAN; loss_G_GAN_Feat += l_G_GAN_Feat
            loss_D_real += l_D_real; loss_D_fake += l_D_fake

        if self.opt.add_face_disc:
            face_weight = 2
            ys, ye, xs, xe = self.get_face_region(real_A)
            if ys is not None:                
                loss_D_f_real, loss_D_f_fake, loss_G_f_GAN, loss_G_f_GAN_Feat = self.compute_loss_D(self.netD_f,
                    real_A[:,:,ys:ye,xs:xe], real_B[:,:,ys:ye,xs:xe], fake_B[:,:,ys:ye,xs:xe])  
                loss_G_f_GAN *= face_weight  
                loss_G_f_GAN_Feat *= face_weight                  
            else:
                loss_D_f_real = loss_D_f_fake = loss_G_f_GAN = loss_G_f_GAN_Feat = torch.zeros_like(loss_D_real)

        loss_list = [loss_G_VGG, loss_G_GAN, loss_G_GAN_Feat,
                     loss_D_real, loss_D_fake, 
                     loss_G_Warp, loss_F_Flow, loss_F_Warp, loss_W]
        if self.opt.add_face_disc:
            loss_list += [loss_G_f_GAN, loss_G_f_GAN_Feat, loss_D_f_real, loss_D_f_fake]   
        loss_list = [loss.unsqueeze(0) for loss in loss_list]           
        return loss_list

    def get_face_region(self, real_A):
        _, _, h, w = real_A.size()
        if not self.opt.openpose_only:
            face = (real_A[:,2] > 0.9).nonzero()
        else:            
            face = ((real_A[:,0] > 0.19) & (real_A[:,0] < 0.21) & (real_A[:,1] < -0.99) & (real_A[:,2] > -0.61) & (real_A[:,2] < -0.59)).nonzero()
        if face.size()[0]:
            y, x = face[:,1], face[:,2]
            ys, ye, xs, xe = y.min().item(), y.max().item(), x.min().item(), x.max().item()
            yc, ylen = int(ys+ye)//2, self.opt.fineSize//32*8
            xc, xlen = int(xs+xe)//2, self.opt.fineSize//32*8
            yc = max(ylen//2, min(h-1 - ylen//2, yc))
            xc = max(xlen//2, min(w-1 - xlen//2, xc))
            ys, ye, xs, xe = yc - ylen//2, yc + ylen//2, xc - xlen//2, xc + xlen//2
            return ys, ye, xs, xe
        return None, None, None, None

    def save(self, label):
        self.save_network(self.netD, 'D', label, self.gpu_ids)         
        for s in range(self.opt.n_scales_temporal):
            self.save_network(getattr(self, 'netD_T'+str(s)), 'D_T'+str(s), label, self.gpu_ids)   
        if self.opt.add_face_disc:
            self.save_network(self.netD_f, 'D_f', label, self.gpu_ids)  
       
    def update_learning_rate(self, epoch):        
        lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr