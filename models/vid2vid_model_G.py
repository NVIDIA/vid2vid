### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import math
import torch
import torch.nn.functional as F
import os
import sys
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks

class Vid2VidModelG(BaseModel):
    def name(self):
        return 'Vid2VidModelG'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain        
        if not opt.debug:
            torch.backends.cudnn.benchmark = True       
        
        # define net G                        
        self.n_scales = opt.n_scales_spatial        
        self.use_single_G = opt.use_single_G
        self.split_gpus = (self.opt.n_gpus_gen < len(self.opt.gpu_ids)) and (self.opt.batchSize == 1)

        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        netG_input_nc = input_nc * opt.n_frames_G
        if opt.use_instance:
            netG_input_nc += opt.n_frames_G        
        prev_output_nc = (opt.n_frames_G - 1) * opt.output_nc 
        if opt.openpose_only:
            opt.no_flow = True     

        self.netG0 = networks.define_G(netG_input_nc, opt.output_nc, prev_output_nc, opt.ngf, opt.netG, 
                                       opt.n_downsample_G, opt.norm, 0, self.gpu_ids, opt)
        for s in range(1, self.n_scales):            
            ngf = opt.ngf // (2**s)
            setattr(self, 'netG'+str(s), networks.define_G(netG_input_nc, opt.output_nc, prev_output_nc, ngf, opt.netG+'Local', 
                                                           opt.n_downsample_G, opt.norm, s, self.gpu_ids, opt))

        print('---------- Networks initialized -------------') 
        print('-----------------------------------------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:                    
            for s in range(self.n_scales):
                self.load_network(getattr(self, 'netG'+str(s)), 'G'+str(s), opt.which_epoch, opt.load_pretrain)
                
        self.netG_i = self.load_single_G() if self.use_single_G else None
        
        # define training variables
        if self.isTrain:            
            self.n_gpus = self.opt.n_gpus_gen if self.opt.batchSize == 1 else 1    # number of gpus for running generator            
            self.n_frames_bp = 1                                                   # number of frames to backpropagate the loss            
            self.n_frames_per_gpu = min(self.opt.max_frames_per_gpu, self.opt.n_frames_total // self.n_gpus) # number of frames in each GPU
            self.n_frames_load = self.n_gpus * self.n_frames_per_gpu   # number of frames in all GPUs            
            if self.opt.debug:
                print('training %d frames at once, using %d gpus, frames per gpu = %d' % (self.n_frames_load, 
                    self.n_gpus, self.n_frames_per_gpu))

        # set loss functions and optimizers
        if self.isTrain:            
            self.old_lr = opt.lr
            self.finetune_all = opt.niter_fix_global == 0
            if not self.finetune_all:
                print('------------ Only updating the finest scale for %d epochs -----------' % opt.niter_fix_global)
          
            # initialize optimizer G
            params = list(getattr(self, 'netG'+str(self.n_scales-1)).parameters())
            if self.finetune_all:
                for s in range(self.n_scales-1):
                    params += list(getattr(self, 'netG'+str(s)).parameters())

            if opt.TTUR:                
                beta1, beta2 = 0, 0.9
                lr = opt.lr / 2
            else:
                beta1, beta2 = opt.beta1, 0.999
                lr = opt.lr            
            self.optimizer_G = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))

    def encode_input(self, input_map, real_image, inst_map=None):        
        size = input_map.size()
        self.bs, tG, self.height, self.width = size[0], size[1], size[3], size[4]
        
        input_map = input_map.data.cuda()                
        if self.opt.label_nc != 0:                        
            # create one-hot vector for label map             
            oneHot_size = (self.bs, tG, self.opt.label_nc, self.height, self.width)
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(2, input_map.long(), 1.0)    
            input_map = input_label        
        input_map = Variable(input_map)
                
        if self.opt.use_instance:
            inst_map = inst_map.data.cuda()            
            edge_map = Variable(self.get_edges(inst_map))            
            input_map = torch.cat([input_map, edge_map], dim=2)
        
        pool_map = None
        if self.opt.dataset_mode == 'face':
            pool_map = inst_map.data.cuda()
        
        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())   

        return input_map, real_image, pool_map

    def forward(self, input_A, input_B, inst_A, fake_B_prev, dummy_bs=0):
        tG = self.opt.n_frames_G           
        gpu_split_id = self.opt.n_gpus_gen + 1        
        if input_A.get_device() == self.gpu_ids[0]:
            input_A, input_B, inst_A, fake_B_prev = util.remove_dummy_from_tensor([input_A, input_B, inst_A, fake_B_prev], dummy_bs)
            if input_A.size(0) == 0: return self.return_dummy(input_A)
        real_A_all, real_B_all, _ = self.encode_input(input_A, input_B, inst_A)        

        is_first_frame = fake_B_prev is None
        if is_first_frame: # at the beginning of a sequence; needs to generate the first frame
            fake_B_prev = self.generate_first_frame(real_A_all, real_B_all)                    
                        
        netG = []
        for s in range(self.n_scales): # broadcast netG to all GPUs used for generator
            netG_s = getattr(self, 'netG'+str(s))                        
            netG_s = torch.nn.parallel.replicate(netG_s, self.opt.gpu_ids[:gpu_split_id]) if self.split_gpus else [netG_s]
            netG.append(netG_s)

        start_gpu = self.gpu_ids[1] if self.split_gpus else real_A_all.get_device()        
        fake_B, fake_B_raw, flow, weight = self.generate_frame_train(netG, real_A_all, fake_B_prev, start_gpu, is_first_frame)        
        fake_B_prev = [B[:, -tG+1:].detach() for B in fake_B]
        fake_B = [B[:, tG-1:] for B in fake_B]

        return fake_B[0], fake_B_raw, flow, weight, real_A_all[:,tG-1:], real_B_all[:,tG-2:], fake_B_prev

    def generate_frame_train(self, netG, real_A_all, fake_B_pyr, start_gpu, is_first_frame):        
        tG = self.opt.n_frames_G        
        n_frames_load = self.n_frames_load
        n_scales = self.n_scales
        finetune_all = self.finetune_all
        dest_id = self.gpu_ids[0] if self.split_gpus else start_gpu        

        ### generate inputs   
        real_A_pyr = self.build_pyr(real_A_all)        
        fake_Bs_raw, flows, weights = None, None, None            
        
        ### sequentially generate each frame
        for t in range(n_frames_load):
            gpu_id = (t // self.n_frames_per_gpu + start_gpu) if self.split_gpus else start_gpu # the GPU idx where we generate this frame
            net_id = gpu_id if self.split_gpus else 0                                           # the GPU idx where the net is located
            fake_B_feat = flow_feat = fake_B_fg_feat = None

            # coarse-to-fine approach
            for s in range(n_scales):
                si = n_scales-1-s
                ### prepare inputs                
                # 1. input labels
                real_As = real_A_pyr[si]
                _, _, _, h, w = real_As.size()                  
                real_As_reshaped = real_As[:, t:t+tG,...].view(self.bs, -1, h, w).cuda(gpu_id)              

                # 2. previous fake_Bs                
                fake_B_prevs = fake_B_pyr[si][:, t:t+tG-1,...].cuda(gpu_id)
                if (t % self.n_frames_bp) == 0:
                    fake_B_prevs = fake_B_prevs.detach()
                fake_B_prevs_reshaped = fake_B_prevs.view(self.bs, -1, h, w)
                
                # 3. mask for foreground and whether to use warped previous image
                mask_F = self.compute_mask(real_As, t+tG-1) if self.opt.fg else None
                use_raw_only = self.opt.no_first_img and is_first_frame 

                ### network forward                                                
                fake_B, flow, weight, fake_B_raw, fake_B_feat, flow_feat, fake_B_fg_feat \
                    = netG[s][net_id].forward(real_As_reshaped, fake_B_prevs_reshaped, mask_F, 
                                              fake_B_feat, flow_feat, fake_B_fg_feat, use_raw_only)

                # if only training the finest scale, leave the coarser levels untouched
                if s != n_scales-1 and not finetune_all:
                    fake_B, fake_B_feat = fake_B.detach(), fake_B_feat.detach()
                    if flow is not None:
                        flow, flow_feat = flow.detach(), flow_feat.detach()
                    if fake_B_fg_feat is not None:
                        fake_B_fg_feat = fake_B_fg_feat.detach()
                
                # collect results into a sequence
                fake_B_pyr[si] = self.concat([fake_B_pyr[si], fake_B.unsqueeze(1).cuda(dest_id)], dim=1)                                
                if s == n_scales-1:                    
                    fake_Bs_raw = self.concat([fake_Bs_raw, fake_B_raw.unsqueeze(1).cuda(dest_id)], dim=1)
                    if flow is not None:
                        flows = self.concat([flows, flow.unsqueeze(1).cuda(dest_id)], dim=1)
                        weights = self.concat([weights, weight.unsqueeze(1).cuda(dest_id)], dim=1)                        
        
        return fake_B_pyr, fake_Bs_raw, flows, weights

    def inference(self, input_A, input_B, inst_A):
        with torch.no_grad():
            real_A, real_B, pool_map = self.encode_input(input_A, input_B, inst_A)            
            self.is_first_frame = not hasattr(self, 'fake_B_prev') or self.fake_B_prev is None
            if self.is_first_frame:
                self.fake_B_prev = self.generate_first_frame(real_A, real_B, pool_map)                 
            
            real_A = self.build_pyr(real_A)            
            self.fake_B_feat = self.flow_feat = self.fake_B_fg_feat = None            
            for s in range(self.n_scales):
                fake_B = self.generate_frame_infer(real_A[self.n_scales-1-s], s)
        return fake_B, real_A[0][0, -1]

    def generate_frame_infer(self, real_A, s):
        tG = self.opt.n_frames_G
        _, _, _, h, w = real_A.size()
        si = self.n_scales-1-s
        netG_s = getattr(self, 'netG'+str(s))
        
        ### prepare inputs
        real_As_reshaped = real_A[0,:tG].view(1, -1, h, w)
        fake_B_prevs_reshaped = self.fake_B_prev[si].view(1, -1, h, w)               
        mask_F = self.compute_mask(real_A, tG-1)[0] if self.opt.fg else None
        use_raw_only = self.opt.no_first_img and self.is_first_frame

        ### network forward        
        fake_B, flow, weight, fake_B_raw, self.fake_B_feat, self.flow_feat, self.fake_B_fg_feat \
            = netG_s.forward(real_As_reshaped, fake_B_prevs_reshaped, mask_F, 
                             self.fake_B_feat, self.flow_feat, self.fake_B_fg_feat, use_raw_only)    

        self.fake_B_prev[si] = torch.cat([self.fake_B_prev[si][1:,...], fake_B])        
        return fake_B

    def generate_first_frame(self, real_A, real_B, pool_map=None):
        tG = self.opt.n_frames_G
        if self.opt.no_first_img:          # model also generates first frame            
            fake_B_prev = Variable(self.Tensor(self.bs, tG-1, self.opt.output_nc, self.height, self.width).zero_())
        elif self.opt.isTrain or self.opt.use_real_img: # assume first frame is given
            fake_B_prev = real_B[:,:(tG-1),...]            
        elif self.opt.use_single_G:        # use another model (trained on single images) to generate first frame
            fake_B_prev = None
            if self.opt.use_instance:
                real_A = real_A[:,:,:self.opt.label_nc,:,:]
            for i in range(tG-1):                
                feat_map = self.get_face_features(real_B[:,i], pool_map[:,i]) if self.opt.dataset_mode == 'face' else None
                fake_B = self.netG_i.forward(real_A[:,i], feat_map).unsqueeze(1)                
                fake_B_prev = self.concat([fake_B_prev, fake_B], dim=1)
        else:
            raise ValueError('Please specify the method for generating the first frame')
            
        fake_B_prev = self.build_pyr(fake_B_prev)
        if not self.opt.isTrain:
            fake_B_prev = [B[0] for B in fake_B_prev]
        return fake_B_prev    

    def return_dummy(self, input_A):
        h, w = input_A.size()[3:]
        t = self.n_frames_load
        tG = self.opt.n_frames_G  
        flow, weight = (self.Tensor(1, t, 2, h, w), self.Tensor(1, t, 1, h, w)) if not self.opt.no_flow else (None, None)
        return self.Tensor(1, t, 3, h, w), self.Tensor(1, t, 3, h, w), flow, weight, \
               self.Tensor(1, t, self.opt.input_nc, h, w), self.Tensor(1, t+1, 3, h, w), self.build_pyr(self.Tensor(1, tG-1, 3, h, w))

    def load_single_G(self): # load the model that generates the first frame
        opt = self.opt     
        s = self.n_scales
        if 'City' in self.opt.dataroot:
            single_path = 'checkpoints/label2city_single/'
            if opt.loadSize == 512:
                load_path = single_path + 'latest_net_G_512.pth'            
                netG = networks.define_G(35, 3, 0, 64, 'global', 3, 'instance', 0, self.gpu_ids, opt)                
            elif opt.loadSize == 1024:                            
                load_path = single_path + 'latest_net_G_1024.pth'
                netG = networks.define_G(35, 3, 0, 64, 'global', 4, 'instance', 0, self.gpu_ids, opt)                
            elif opt.loadSize == 2048:     
                load_path = single_path + 'latest_net_G_2048.pth'
                netG = networks.define_G(35, 3, 0, 32, 'local', 4, 'instance', 0, self.gpu_ids, opt)
            else:
                raise ValueError('Single image generator does not exist')
        elif 'face' in self.opt.dataroot:            
            single_path = 'checkpoints/edge2face_single/'
            load_path = single_path + 'latest_net_G.pth' 
            opt.feat_num = 16           
            netG = networks.define_G(15, 3, 0, 64, 'global_with_features', 3, 'instance', 0, self.gpu_ids, opt)
            encoder_path = single_path + 'latest_net_E.pth'
            self.netE = networks.define_G(3, 16, 0, 16, 'encoder', 4, 'instance', 0, self.gpu_ids)
            self.netE.load_state_dict(torch.load(encoder_path))
        else:
            raise ValueError('Single image generator does not exist')
        netG.load_state_dict(torch.load(load_path))        
        return netG

    def get_face_features(self, real_image, inst):                
        feat_map = self.netE.forward(real_image, inst)            
        #if self.opt.use_encoded_image:
        #    return feat_map
        
        load_name = 'checkpoints/edge2face_single/features.npy'
        features = np.load(load_name, encoding='latin1').item()                        
        inst_np = inst.cpu().numpy().astype(int)

        # find nearest neighbor in the training dataset
        num_images = features[6].shape[0]
        feat_map = feat_map.data.cpu().numpy()
        feat_ori = torch.FloatTensor(7, self.opt.feat_num, 1) # feature map for test img (for each facial part)
        feat_ref = torch.FloatTensor(7, self.opt.feat_num, num_images) # feature map for training imgs
        for label in np.unique(inst_np):
            idx = (inst == int(label)).nonzero() 
            for k in range(self.opt.feat_num): 
                feat_ori[label,k] = float(feat_map[idx[0,0], idx[0,1] + k, idx[0,2], idx[0,3]])
                for m in range(num_images):
                    feat_ref[label,k,m] = features[label][m,k]                
        cluster_idx = self.dists_min(feat_ori.expand_as(feat_ref).cuda(), feat_ref.cuda(), num=1)

        # construct new feature map from nearest neighbors
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for label in np.unique(inst_np):
            feat = features[label][:,:-1]                                                    
            idx = (inst == int(label)).nonzero()                
            for k in range(self.opt.feat_num):                    
                feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[min(cluster_idx, feat.shape[0]-1), k]
        
        return Variable(feat_map)

    def compute_mask(self, real_As, ts, te=None): # compute the mask for foreground objects
        _, _, _, h, w = real_As.size() 
        if te is None:
            te = ts + 1        
        mask_F = real_As[:, ts:te, self.opt.fg_labels[0]].clone()
        for i in range(1, len(self.opt.fg_labels)):
            mask_F = mask_F + real_As[:, ts:te, self.opt.fg_labels[i]]
        mask_F = torch.clamp(mask_F, 0, 1)
        return mask_F    

    def compute_fake_B_prev(self, real_B_prev, fake_B_last, fake_B):
        fake_B_prev = real_B_prev[:, 0:1] if fake_B_last is None else fake_B_last[0][:, -1:]
        if fake_B.size()[1] > 1:
            fake_B_prev = torch.cat([fake_B_prev, fake_B[:, :-1].detach()], dim=1)
        return fake_B_prev

    def save(self, label):        
        for s in range(self.n_scales):
            self.save_network(getattr(self, 'netG'+str(s)), 'G'+str(s), label, self.gpu_ids)                    