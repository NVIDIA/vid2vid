import os.path
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np

from data.base_dataset import BaseDataset, get_img_params, get_transform, get_video_params, concat_frame
from data.image_folder import make_grouped_dataset, check_path_valid
from data.keypoint2img import read_keypoints

class PoseDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot 

        self.dir_dp = os.path.join(opt.dataroot, opt.phase + '_densepose')
        self.dir_op = os.path.join(opt.dataroot, opt.phase + '_openpose')
        self.dir_img = os.path.join(opt.dataroot, opt.phase + '_img')                
        self.img_paths = sorted(make_grouped_dataset(self.dir_img))
        if not opt.openpose_only:
            self.dp_paths = sorted(make_grouped_dataset(self.dir_dp))
            check_path_valid(self.dp_paths, self.img_paths)
        if not opt.densepose_only:
            self.op_paths = sorted(make_grouped_dataset(self.dir_op))                
            check_path_valid(self.op_paths, self.img_paths)

        self.init_frame_idx(self.img_paths)

    def __getitem__(self, index):
        A, B, _, seq_idx = self.update_frame_idx(self.img_paths, index)
        img_paths = self.img_paths[seq_idx]        
        n_frames_total, start_idx, t_step = get_video_params(self.opt, self.n_frames_total, len(img_paths), self.frame_idx)
        
        img = Image.open(img_paths[start_idx]).convert('RGB')     
        size = img.size
        params = get_img_params(self.opt, size)

        frame_range = list(range(n_frames_total)) if (self.opt.isTrain or self.A is None) else [self.opt.n_frames_G-1]
        for i in frame_range:
            img_path = img_paths[start_idx + i * t_step]
            if not self.opt.openpose_only:
                dp_path = self.dp_paths[seq_idx][start_idx + i * t_step]
                Di = self.get_image(dp_path, size, params, input_type='densepose')
                Di[2,:,:] = ((Di[2,:,:] * 0.5 + 0.5) * 255 / 24 - 0.5) / 0.5
            if not self.opt.densepose_only:
                op_path = self.op_paths[seq_idx][start_idx + i * t_step]
                Oi = self.get_image(op_path, size, params, input_type='openpose')

            if self.opt.openpose_only:
                Ai = Oi
            elif self.opt.densepose_only:
                Ai = Di
            else:
                Ai = torch.cat([Di, Oi])
            Bi = self.get_image(img_path, size, params, input_type='img')
            
            Ai, Bi = self.crop(Ai), self.crop(Bi) # only crop the central half region to save time
            A = concat_frame(A, Ai, n_frames_total)
            B = concat_frame(B, Bi, n_frames_total)
        
        if not self.opt.isTrain:
            self.A, self.B = A, B
            self.frame_idx += 1            
        change_seq = False if self.opt.isTrain else self.change_seq
        return_list = {'A': A, 'B': B, 'inst': 0, 'A_path': img_path, 'change_seq': change_seq}

        return return_list

    def get_image(self, A_path, size, params, input_type):
        if input_type != 'openpose':
            A_img = Image.open(A_path).convert('RGB')
        else:            
            random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0
            A_img = Image.fromarray(read_keypoints(A_path, size, random_drop_prob, self.opt.remove_face_labels, self.opt.basic_point_only))            

        if input_type == 'densepose' and self.opt.isTrain:
            # randomly remove labels
            A_np = np.array(A_img)
            part_labels = A_np[:,:,2]            
            for part_id in range(1, 25):
                if (np.random.rand() < self.opt.random_drop_prob):
                    A_np[(part_labels == part_id), :] = 0
            if self.opt.remove_face_labels:            
                A_np[(part_labels == 23) | (part_labels == 24), :] = 0
            A_img = Image.fromarray(A_np)

        is_img = input_type == 'img'
        method = Image.BICUBIC if is_img else Image.NEAREST
        transform_scaleA = get_transform(self.opt, params, method=method)
        A_scaled = transform_scaleA(A_img)
        return A_scaled

    def crop(self, Ai):
        w = Ai.size()[2]
        base = 32
        x_cen = w // 2
        bs = int(w * 0.25) // base * base
        return Ai[:,:,(x_cen-bs):(x_cen+bs)]
               
    def normalize_pose(self, A_img, target_yc, target_len, first=False):
        w, h = A_img.size
        A_np = np.array(A_img)  

        if first == True:          
            part_labels = A_np[:,:,2]            
            part_coords = np.nonzero((part_labels == 1) | (part_labels == 2))
            y, x = part_coords[0], part_coords[1]

            ys, ye = y.min(), y.max()                    
            min_i, max_i = np.argmin(y), np.argmax(y)
            v_min = A_np[y[min_i], x[min_i], 1] / 255
            v_max = A_np[y[max_i], x[max_i], 1] / 255
            ylen = (ye-ys) / (v_max-v_min)
            yc = (0.5-v_min) / (v_max-v_min) * (ye-ys) + ys            
            
            ratio = target_len / ylen
            offset_y = int(yc - (target_yc / ratio))
            offset_x = int(w * (1 - 1/ratio) / 2)        

            padding = int(max(0, max(-offset_y, int(offset_y + h/ratio) - h)))
            padding = int(max(padding, max(-offset_x, int(offset_x + w/ratio) - w)))
            offset_y += padding
            offset_x += padding            
            self.offset_y, self.offset_x = offset_y, offset_x
            self.ratio, self.padding = ratio, padding

        p = self.padding
        A_np = np.pad(A_np, ((p,p),(p,p),(0,0)), 'constant', constant_values=0)
        A_np = A_np[self.offset_y:int(self.offset_y + h/self.ratio), self.offset_x:int(self.offset_x + w/self.ratio):, :]        
        A_img = Image.fromarray(A_np)
        A_img = A_img.resize((w, h))
        return A_img

    def __len__(self):        
        return sum(self.frames_count)

    def name(self):
        return 'PoseDataset'

"""
DensePose label
0      = Background
1, 2   = Torso
3      = Right Hand
4      = Left Hand
5      = Right Foot
6      = Left Foot
7, 9   = Upper Leg Right
8, 10  = Upper Leg Left
11, 13 = Lower Leg Right
12, 14 = Lower Leg Left
15, 17 = Upper Arm Left
16, 18 = Upper Arm Right
19, 21 = Lower Arm Left
20, 22 = Lower Arm Right
23, 24 = Head """
