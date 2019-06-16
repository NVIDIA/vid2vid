from util.util import add_dummy_to_tensor
import torch.utils.data as data
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

    def update_training_batch(self, ratio): # update the training sequence length to be longer      
        seq_len_max = min(128, self.seq_len_max) - (self.opt.n_frames_G - 1)
        if self.n_frames_total < seq_len_max:
            self.n_frames_total = min(seq_len_max, self.opt.n_frames_total * (2**ratio))
            #self.n_frames_total = min(seq_len_max, self.opt.n_frames_total * (ratio + 1))
            print('--------- Updating training sequence length to %d ---------' % self.n_frames_total)

    def init_frame_idx(self, A_paths):
        self.n_of_seqs = min(len(A_paths), self.opt.max_dataset_size)         # number of sequences to train
        self.seq_len_max = max([len(A) for A in A_paths])                     # max number of frames in the training sequences

        self.seq_idx = 0                                                      # index for current sequence
        self.frame_idx = self.opt.start_frame if not self.opt.isTrain else 0  # index for current frame in the sequence
        self.frames_count = []                                                # number of frames in each sequence
        for path in A_paths:
            self.frames_count.append(len(path) - self.opt.n_frames_G + 1)

        self.folder_prob = [count / sum(self.frames_count) for count in self.frames_count]
        self.n_frames_total = self.opt.n_frames_total if self.opt.isTrain else 1 
        self.A, self.B, self.I = None, None, None

    def update_frame_idx(self, A_paths, index):
        if self.opt.isTrain:
            if self.opt.dataset_mode == 'pose':                
                seq_idx = np.random.choice(len(A_paths), p=self.folder_prob) # randomly pick sequence to train
                self.frame_idx = index
            else:    
                seq_idx = index % self.n_of_seqs            
            return None, None, None, seq_idx
        else:
            self.change_seq = self.frame_idx >= self.frames_count[self.seq_idx]
            if self.change_seq:
                self.seq_idx += 1
                self.frame_idx = 0
                self.A, self.B, self.I = None, None, None
            return self.A, self.B, self.I, self.seq_idx

    def init_data_params(self, data, n_gpus, tG):
        opt = self.opt
        _, n_frames_total, self.height, self.width = data['B'].size()  # n_frames_total = n_frames_load * n_loadings + tG - 1        
        n_frames_total = n_frames_total // opt.output_nc
        n_frames_load = opt.max_frames_per_gpu * n_gpus                # number of total frames loaded into GPU at a time for each batch
        n_frames_load = min(n_frames_load, n_frames_total - tG + 1)
        self.t_len = n_frames_load + tG - 1                             # number of loaded frames plus previous frames
        return n_frames_total-self.t_len+1, n_frames_load, self.t_len

    def init_data(self, t_scales):
        fake_B_last = None  # the last generated frame from previous training batch (which becomes input to the next batch)
        real_B_all, fake_B_all, flow_ref_all, conf_ref_all = None, None, None, None # all real/generated frames so far
        if self.opt.sparse_D:
            real_B_all, fake_B_all, flow_ref_all, conf_ref_all = [None]*t_scales, [None]*t_scales, [None]*t_scales, [None]*t_scales
        
        frames_all = real_B_all, fake_B_all, flow_ref_all, conf_ref_all        
        return fake_B_last, frames_all

    def prepare_data(self, data, i, input_nc, output_nc):
        t_len, height, width = self.t_len, self.height, self.width
        # 5D tensor: batchSize, # of frames, # of channels, height, width
        input_A = (data['A'][:, i*input_nc:(i+t_len)*input_nc, ...]).view(-1, t_len, input_nc, height, width)
        input_B = (data['B'][:, i*output_nc:(i+t_len)*output_nc, ...]).view(-1, t_len, output_nc, height, width)                
        inst_A = (data['inst'][:, i:i+t_len, ...]).view(-1, t_len, 1, height, width) if len(data['inst'].size()) > 2 else None
        return [input_A, input_B, inst_A]

def make_power_2(n, base=32.0):    
    return int(round(n / base) * base)

def get_img_params(opt, size):
    w, h = size
    new_h, new_w = h, w        
    if 'resize' in opt.resize_or_crop:   # resize image to be loadSize x loadSize
        new_h = new_w = opt.loadSize            
    elif 'scaleWidth' in opt.resize_or_crop: # scale image width to be loadSize
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w
    elif 'scaleHeight' in opt.resize_or_crop: # scale image height to be loadSize
        new_h = opt.loadSize
        new_w = opt.loadSize * w // h
    elif 'randomScaleWidth' in opt.resize_or_crop:  # randomly scale image width to be somewhere between loadSize and fineSize
        new_w = random.randint(opt.fineSize, opt.loadSize + 1)
        new_h = new_w * h // w
    elif 'randomScaleHeight' in opt.resize_or_crop: # randomly scale image height to be somewhere between loadSize and fineSize
        new_h = random.randint(opt.fineSize, opt.loadSize + 1)
        new_w = new_h * w // h
    new_w = int(round(new_w / 4)) * 4
    new_h = int(round(new_h / 4)) * 4    

    crop_x = crop_y = 0
    crop_w = crop_h = 0
    if 'crop' in opt.resize_or_crop or 'scaledCrop' in opt.resize_or_crop:
        if 'crop' in opt.resize_or_crop:      # crop patches of size fineSize x fineSize
            crop_w = crop_h = opt.fineSize
        else:
            if 'Width' in opt.resize_or_crop: # crop patches of width fineSize
                crop_w = opt.fineSize
                crop_h = opt.fineSize * h // w
            else:                              # crop patches of height fineSize
                crop_h = opt.fineSize
                crop_w = opt.fineSize * w // h

        crop_w, crop_h = make_power_2(crop_w), make_power_2(crop_h)        
        x_span = (new_w - crop_w) // 2
        crop_x = np.maximum(0, np.minimum(x_span*2, int(np.random.randn() * x_span/3 + x_span)))        
        crop_y = random.randint(0, np.minimum(np.maximum(0, new_h - crop_h), new_h // 8))
        #crop_x = random.randint(0, np.maximum(0, new_w - crop_w))
        #crop_y = random.randint(0, np.maximum(0, new_h - crop_h))        
    else:
        new_w, new_h = make_power_2(new_w), make_power_2(new_h)

    flip = (random.random() > 0.5) and (opt.dataset_mode != 'pose')
    return {'new_size': (new_w, new_h), 'crop_size': (crop_w, crop_h), 'crop_pos': (crop_x, crop_y), 'flip': flip}

def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    ### resize input image
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))   
    else:
        transform_list.append(transforms.Lambda(lambda img: __scale_image(img, params['new_size'], method)))
        
    ### crop patches from image
    if 'crop' in opt.resize_or_crop or 'scaledCrop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_size'], params['crop_pos'])))    

    ### random flip
    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def toTensor_normalize():    
    transform_list = [transforms.ToTensor()]    
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_image(img, size, method=Image.BICUBIC):
    w, h = size    
    return img.resize((w, h), method)

def __crop(img, size, pos):
    ow, oh = img.size
    tw, th = size
    x1, y1 = pos        
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, min(ow, x1 + tw), min(oh, y1 + th)))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def get_video_params(opt, n_frames_total, cur_seq_len, index):
    tG = opt.n_frames_G
    if opt.isTrain:        
        n_frames_total = min(n_frames_total, cur_seq_len - tG + 1)

        n_gpus = opt.n_gpus_gen if opt.batchSize == 1 else 1       # number of generator GPUs for each batch
        n_frames_per_load = opt.max_frames_per_gpu * n_gpus        # number of frames to load into GPUs at one time (for each batch)
        n_frames_per_load = min(n_frames_total, n_frames_per_load)
        n_loadings = n_frames_total // n_frames_per_load           # how many times are needed to load entire sequence into GPUs         
        n_frames_total = n_frames_per_load * n_loadings + tG - 1   # rounded overall number of frames to read from the sequence
        
        max_t_step = min(opt.max_t_step, (cur_seq_len-1) // (n_frames_total-1))
        t_step = np.random.randint(max_t_step) + 1                    # spacing between neighboring sampled frames
        offset_max = max(1, cur_seq_len - (n_frames_total-1)*t_step)  # maximum possible index for the first frame        
        if opt.dataset_mode == 'pose':
            start_idx = index % offset_max
        else:
            start_idx = np.random.randint(offset_max)                 # offset for the first frame to load
        if opt.debug:
            print("loading %d frames in total, first frame starting at index %d, space between neighboring frames is %d"
                % (n_frames_total, start_idx, t_step))
    else:
        n_frames_total = tG
        start_idx = index
        t_step = 1   
    return n_frames_total, start_idx, t_step

def concat_frame(A, Ai, nF):
    if A is None:
        A = Ai
    else:
        c = Ai.size()[0]
        if A.size()[0] == nF * c:
            A = A[c:]
        A = torch.cat([A, Ai])
    return A