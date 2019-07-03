import os, sys
import numpy as np
import torch
from .networks import get_grid

class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    def resolve_version(self):
        import torch._utils
        try:
            torch._utils._rebuild_tensor_v2
        except AttributeError:
            def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
                tensor.requires_grad = requires_grad
                tensor._backward_hooks = backward_hooks
                return tensor
            torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):        
        self.resolve_version()    
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)        
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if 'G0' in network_label:
                raise('Generator must exist!')
        else:
            #network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path))
            except:   
                pretrained_dict = torch.load(save_path)                
                model_dict = network.state_dict()

                ### printout layers in pretrained model
                initialized = set()                    
                for k, v in pretrained_dict.items():                      
                    initialized.add(k.split('.')[0])                         
                #print('pretrained model has following layers: ')
                #print(sorted(initialized))                

                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                    network.load_state_dict(pretrained_dict)
                    print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    if sys.version_info >= (3,0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()
                    for k, v in pretrained_dict.items():                      
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])                            
                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)                  

    def concat(self, tensors, dim=0):
        if tensors[0] is not None and tensors[1] is not None:
            if isinstance(tensors[0], list):                
                tensors_cat = []
                for i in range(len(tensors[0])):                    
                    tensors_cat.append(self.concat([tensors[0][i], tensors[1][i]], dim=dim))                
                return tensors_cat
            return torch.cat([tensors[0], tensors[1]], dim=dim)
        elif tensors[0] is not None:
            return tensors[0]
        else:
            return tensors[1]

    def build_pyr(self, tensor, nearest=False): # build image pyramid from a single image
        if tensor is None:
            return [None] * self.n_scales
        tensor = [tensor]
        if nearest:
            downsample = torch.nn.AvgPool2d(1, stride=2)
        else:
            downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)        
        for s in range(1, self.n_scales):
            b, t, c, h, w = tensor[-1].size()
            down = downsample(tensor[-1].view(-1, h, w)).view(b, t, c, h//2, w//2)
            tensor.append(down)
        return tensor

    def dists_min(self, a, b, num=1):        
        dists = torch.sum(torch.sum((a-b)*(a-b), dim=0), dim=0)        
        if num == 1:
            val, idx = torch.min(dists, dim=0)        
            #idx = [idx]
        else:
            val, idx = torch.sort(dists, dim=0)
            idx = idx[:num]
        return idx.cpu().numpy().astype(int)

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,:,1:] = edge[:,:,:,:,1:] | (t[:,:,:,:,1:] != t[:,:,:,:,:-1])
        edge[:,:,:,:,:-1] = edge[:,:,:,:,:-1] | (t[:,:,:,:,1:] != t[:,:,:,:,:-1])
        edge[:,:,:,1:,:] = edge[:,:,:,1:,:] | (t[:,:,:,1:,:] != t[:,:,:,:-1,:])
        edge[:,:,:,:-1,:] = edge[:,:,:,:-1,:] | (t[:,:,:,1:,:] != t[:,:,:,:-1,:])
        return edge.float()       
        
    def update_learning_rate(self, epoch, model):        
        lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
        for param_group in getattr(self, 'optimizer_' + model).param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def update_fixed_params(self): # finetune all scales instead of just finest scale
        params = []
        for s in range(self.n_scales):
            params += list(getattr(self, 'netG'+str(s)).parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.old_lr, betas=(self.opt.beta1, 0.999))
        self.finetune_all = True
        print('------------ Now finetuning all scales -----------')

    def update_training_batch(self, ratio): # increase number of backpropagated frames and number of frames in each GPU
        nfb = self.n_frames_bp
        nfl = self.n_frames_load
        if nfb < nfl:            
            nfb = min(self.opt.max_frames_backpropagate, 2**ratio)
            self.n_frames_bp = nfl // int(np.ceil(float(nfl) / nfb))
            print('-------- Updating number of backpropagated frames to %d ----------' % self.n_frames_bp)

        if self.n_frames_per_gpu < self.opt.max_frames_per_gpu:
            self.n_frames_per_gpu = min(self.n_frames_per_gpu*2, self.opt.max_frames_per_gpu)
            self.n_frames_load = self.n_gpus * self.n_frames_per_gpu
            print('-------- Updating number of frames per gpu to %d ----------' % self.n_frames_per_gpu)


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