import torch.utils.data as data
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

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if 'resize' in opt.resize_or_crop:
        new_h = new_w = opt.loadSize            
    elif 'scaleWidth' in opt.resize_or_crop:
        new_w = opt.loadSize
        new_h = opt.loadSize * h / w

    if 'crop' in opt.resize_or_crop:
        x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
        y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    elif 'scaledCrop' in opt.resize_or_crop:
        x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
        y = random.randint(0, np.maximum(0, new_h - opt.fineSize*new_h//new_w))
    else:
        x = y = 0
    
    flip = random.random() > 0.5
    return {'crop_pos': (x,y), 'flip': flip}

def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))   
    elif 'scaleWidth' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_image(img, opt.loadSize, method)))
        
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))
    elif 'scaledCrop' in opt.resize_or_crop:        
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize, False)))
        
    elif opt.resize_or_crop == 'none':
        base = 32        
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))    

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

def __scale_image(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow > oh:
        w = target_width
        h = int(target_width * oh / ow)        
    else:
        h = target_width
        w = int(target_width * ow / oh)
    base = 32.0
    h = int(round(h / base) * base)
    w = int(round(w / base) * base)
    return img.resize((w, h), method)

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    base = 32.0
    h = int(round(h / base) * base)
    w = int(round(w / base) * base)
    return img.resize((w, h), method)


def __crop(img, pos, size, square=True):
    ow, oh = img.size
    x1, y1 = pos
    tw, th = size
    if not square:
        th = th * oh // ow    
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, min(ow, x1 + tw), min(oh, y1 + th)))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
