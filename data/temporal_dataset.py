### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_grouped_dataset
from PIL import Image
import numpy as np

class TemporalDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')
        self.A_is_label = self.opt.label_nc != 0

        self.A_paths = sorted(make_grouped_dataset(self.dir_A))
        self.B_paths = sorted(make_grouped_dataset(self.dir_B))
        assert(len(self.A_paths) == len(self.B_paths))
        if opt.use_instance:                
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.I_paths = sorted(make_grouped_dataset(self.dir_inst))
            assert(len(self.A_paths) == len(self.I_paths))

        self.n_of_seqs = len(self.A_paths)                 # number of sequences to train       
        self.seq_len_max = len(self.A_paths[0])            # max number of frames in the training sequences
        for i in range(1, self.n_of_seqs):
            self.seq_len_max = max(self.seq_len_max, len(self.A_paths[i]))        
        self.n_frames_total = self.opt.n_frames_total      # current number of frames to train in a single iteration

    def __getitem__(self, index):
        tG = self.opt.n_frames_G        
        A_paths = self.A_paths[index % self.n_of_seqs]
        B_paths = self.B_paths[index % self.n_of_seqs]        
        assert(len(A_paths) == len(B_paths))
        if self.opt.use_instance:
            I_paths = self.I_paths[index % self.n_of_seqs]            
            assert(len(A_paths) == len(I_paths))
        
        # setting parameters
        cur_seq_len = len(A_paths)
        n_frames_total = min(self.n_frames_total, cur_seq_len - tG + 1)

        n_gpus = self.opt.n_gpus_gen // self.opt.batchSize         # number of generator GPUs for each batch
        n_frames_per_load = self.opt.max_frames_per_gpu * n_gpus   # number of frames to load into GPUs at one time (for each batch)
        n_frames_per_load = min(n_frames_total, n_frames_per_load)
        n_loadings = n_frames_total // n_frames_per_load           # how many times are needed to load entire sequence into GPUs         
        n_frames_total = n_frames_per_load * n_loadings + tG - 1   # rounded overall number of frames to read from the sequence

        #t_step_max = min(1, (cur_seq_len-1) // (n_frames_total-1))
        #t_step = np.random.randint(t_step_max) + 1                   # spacing between neighboring sampled frames
        t_step = 1
        offset_max = max(1, cur_seq_len - (n_frames_total-1)*t_step)  # maximum possible index for the first frame
        start_idx = np.random.randint(offset_max)                     # offset for the first frame to load        
        if self.opt.debug:
            print("loading %d frames in total, first frame starting at index %d" % (n_frames_total, start_idx))

        # setting transformers
        B_img = Image.open(B_paths[0]).convert('RGB')        
        params = get_params(self.opt, B_img.size)          
        transform_scaleB = get_transform(self.opt, params)
        transform_scaleA = get_transform(self.opt, params, method=Image.NEAREST, normalize=False) if self.A_is_label else transform_scaleB

        # read in images
        inst = 0
        for i in range(n_frames_total):            
            A_path = A_paths[start_idx + i * t_step]
            B_path = B_paths[start_idx + i * t_step]            
            Ai = self.get_image(A_path, transform_scaleA)
            if self.A_is_label:
                Ai = Ai * 255.0  
            Bi = self.get_image(B_path, transform_scaleB)
            
            A = Ai if i == 0 else torch.cat([A, Ai], dim=0)            
            B = Bi if i == 0 else torch.cat([B, Bi], dim=0)            

            if self.opt.use_instance:
                I_path = I_paths[start_idx + i * t_step]                
                Ii = self.get_image(I_path, transform_scaleA) * 255.0
                inst = Ii if i == 0 else torch.cat([inst, Ii], dim=0)                

        return_list = {'A': A, 'B': B, 'inst': inst, 'A_paths': A_path, 'B_paths': B_path}
        return return_list

    def get_image(self, A_path, transform_scaleA):
        A_img = Image.open(A_path)        
        A_scaled = transform_scaleA(A_img)        
        return A_scaled

    def update_training_batch(self, ratio): # update the training sequence length to be longer      
        seq_len_max = min(128, self.seq_len_max) - (self.opt.n_frames_G - 1)
        if self.n_frames_total < seq_len_max:
            self.n_frames_total = min(seq_len_max, self.opt.n_frames_total * (2**ratio))
            #self.n_frames_total = min(seq_len_max, self.opt.n_frames_total * (ratio + 1))
            print('--------- Updating training sequence length to %d ---------' % self.n_frames_total)

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'TemporalDataset'