### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_grouped_dataset
from PIL import Image
import numpy as np

class TestDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = opt.dataroot
        self.dir_B = opt.dataroot.replace('test_A', 'test_B')
        self.use_real = opt.use_real_img
        self.A_is_label = self.opt.label_nc != 0

        self.A_paths = sorted(make_grouped_dataset(self.dir_A))
        if self.use_real:
            self.B_paths = sorted(make_grouped_dataset(self.dir_B))
            assert(len(self.A_paths) == len(self.B_paths))
        if self.opt.use_instance:                
            self.dir_inst = opt.dataroot.replace('test_A', 'test_inst')
            self.I_paths = sorted(make_grouped_dataset(self.dir_inst))
            assert(len(self.A_paths) == len(self.I_paths))

        self.seq_idx = 0
        self.frame_idx = 0                
        self.frames_count = []
        for path in self.A_paths:
            self.frames_count.append(len(path) - opt.n_frames_G + 1)        

    def __getitem__(self, index):
        tG = self.opt.n_frames_G
        change_seq = self.frame_idx >= self.frames_count[self.seq_idx]
        if change_seq:
            self.seq_idx += 1
            self.frame_idx = 0        
              
        A_img = Image.open(self.A_paths[self.seq_idx][0]).convert('RGB')        
        params = get_params(self.opt, A_img.size)
        transform_scaleB = get_transform(self.opt, params)
        transform_scaleA = get_transform(self.opt, params, method=Image.NEAREST, normalize=False) if self.A_is_label else transform_scaleB
   
        B = inst = 0
        for i in range(tG):                                                   
            A_path = self.A_paths[self.seq_idx][self.frame_idx + i]            
            Ai = self.get_image(A_path, transform_scaleA)            
            if self.A_is_label:
                Ai = Ai * 255.0            
            A = Ai if i == 0 else torch.cat([A, Ai], dim=0)                        

            if self.use_real:
                B_path = self.B_paths[self.seq_idx][self.frame_idx + i]
                Bi = self.get_image(B_path, transform_scaleB)
                B = Bi if i == 0 else torch.cat([B, Bi], dim=0)

            if self.opt.use_instance:
                I_path = self.I_paths[self.seq_idx][self.frame_idx + i]
                Ii = self.get_image(I_path, transform_scaleA) * 255.0
                inst = Ii if i == 0 else torch.cat([inst, Ii], dim=0)

        self.frame_idx += 1
        return_list = {'A': A, 'B': B, 'inst': inst, 'A_paths': A_path, 'change_seq': change_seq}
        return return_list

    def get_image(self, A_path, transform_scaleA):
        A_img = Image.open(A_path)
        A_scaled = transform_scaleA(A_img)        
        return A_scaled

    def __len__(self):        
        return sum(self.frames_count)

    def n_of_seqs(self):        
        return len(self.A_paths)

    def name(self):
        return 'TestDataset'