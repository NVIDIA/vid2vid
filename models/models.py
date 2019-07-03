### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import torch
import torch.nn as nn
import numpy as np
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

def wrap_model(opt, modelG, modelD, flowNet):
    if opt.n_gpus_gen == len(opt.gpu_ids):
        modelG = myModel(opt, modelG)
        modelD = myModel(opt, modelD)
        flowNet = myModel(opt, flowNet)
    else:             
        if opt.batchSize == 1:
            gpu_split_id = opt.n_gpus_gen + 1
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[0:1])                
        else:
            gpu_split_id = opt.n_gpus_gen
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[:gpu_split_id])
        modelD = nn.DataParallel(modelD, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
        flowNet = nn.DataParallel(flowNet, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
    return modelG, modelD, flowNet

class myModel(nn.Module):
    def __init__(self, opt, model):        
        super(myModel, self).__init__()
        self.opt = opt
        self.module = model
        self.model = nn.DataParallel(model, device_ids=opt.gpu_ids)
        self.bs_per_gpu = int(np.ceil(float(opt.batchSize) / len(opt.gpu_ids))) # batch size for each GPU
        self.pad_bs = self.bs_per_gpu * len(opt.gpu_ids) - opt.batchSize           

    def forward(self, *inputs, **kwargs):
        inputs = self.add_dummy_to_tensor(inputs, self.pad_bs)
        outputs = self.model(*inputs, **kwargs, dummy_bs=self.pad_bs)
        if self.pad_bs == self.bs_per_gpu: # gpu 0 does 0 batch but still returns 1 batch
            return self.remove_dummy_from_tensor(outputs, 1)
        return outputs        

    def add_dummy_to_tensor(self, tensors, add_size=0):        
        if add_size == 0 or tensors is None: return tensors
        if type(tensors) == list or type(tensors) == tuple:
            return [self.add_dummy_to_tensor(tensor, add_size) for tensor in tensors]    
                
        if isinstance(tensors, torch.Tensor):            
            dummy = torch.zeros_like(tensors)[:add_size]
            tensors = torch.cat([dummy, tensors])
        return tensors

    def remove_dummy_from_tensor(self, tensors, remove_size=0):
        if remove_size == 0 or tensors is None: return tensors
        if type(tensors) == list or type(tensors) == tuple:
            return [self.remove_dummy_from_tensor(tensor, remove_size) for tensor in tensors]    
        
        if isinstance(tensors, torch.Tensor):
            tensors = tensors[remove_size:]
        return tensors

def create_model(opt):    
    print(opt.model)            
    if opt.model == 'vid2vid':
        from .vid2vid_model_G import Vid2VidModelG
        modelG = Vid2VidModelG()    
        if opt.isTrain:
            from .vid2vid_model_D import Vid2VidModelD
            modelD = Vid2VidModelD()    
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    if opt.isTrain:
        from .flownet import FlowNet
        flowNet = FlowNet()
    
    modelG.initialize(opt)
    if opt.isTrain:
        modelD.initialize(opt)
        flowNet.initialize(opt)        
        if not opt.fp16:
            modelG, modelD, flownet = wrap_model(opt, modelG, modelD, flowNet)
        return [modelG, modelD, flowNet]
    else:
        return modelG

def create_optimizer(opt, models):
    modelG, modelD, flowNet = models
    optimizer_D_T = []    
    if opt.fp16:              
        from apex import amp
        for s in range(opt.n_scales_temporal):
            optimizer_D_T.append(getattr(modelD, 'optimizer_D_T'+str(s)))
        modelG, optimizer_G = amp.initialize(modelG, modelG.optimizer_G, opt_level='O1')
        modelD, optimizers_D = amp.initialize(modelD, [modelD.optimizer_D] + optimizer_D_T, opt_level='O1')
        optimizer_D, optimizer_D_T = optimizers_D[0], optimizers_D[1:]        
        modelG, modelD, flownet = wrap_model(opt, modelG, modelD, flowNet)
    else:        
        optimizer_G = modelG.module.optimizer_G
        optimizer_D = modelD.module.optimizer_D        
        for s in range(opt.n_scales_temporal):
            optimizer_D_T.append(getattr(modelD.module, 'optimizer_D_T'+str(s)))
    return modelG, modelD, flowNet, optimizer_G, optimizer_D, optimizer_D_T

def init_params(opt, modelG, modelD, data_loader):
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    start_epoch, epoch_iter = 1, 0
    ### if continue training, recover previous states
    if opt.continue_train:        
        if os.path.exists(iter_path):
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)        
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))   
        if start_epoch > opt.niter:
            modelG.module.update_learning_rate(start_epoch-1, 'G')
            modelD.module.update_learning_rate(start_epoch-1, 'D')
        if (opt.n_scales_spatial > 1) and (opt.niter_fix_global != 0) and (start_epoch > opt.niter_fix_global):
            modelG.module.update_fixed_params()
        if start_epoch > opt.niter_step:
            data_loader.dataset.update_training_batch((start_epoch-1)//opt.niter_step)
            modelG.module.update_training_batch((start_epoch-1)//opt.niter_step)    

    n_gpus = opt.n_gpus_gen if opt.batchSize == 1 else 1   # number of gpus used for generator for each batch
    tG, tD = opt.n_frames_G, opt.n_frames_D
    tDB = tD * opt.output_nc        
    s_scales = opt.n_scales_spatial
    t_scales = opt.n_scales_temporal
    input_nc = 1 if opt.label_nc != 0 else opt.input_nc
    output_nc = opt.output_nc         

    print_freq = lcm(opt.print_freq, opt.batchSize)
    total_steps = (start_epoch-1) * len(data_loader) + epoch_iter
    total_steps = total_steps // print_freq * print_freq  

    return n_gpus, tG, tD, tDB, s_scales, t_scales, input_nc, output_nc, start_epoch, epoch_iter, print_freq, total_steps, iter_path

def save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD, end_of_epoch=False):
    if not end_of_epoch:
        if total_steps % opt.save_latest_freq == 0:
            visualizer.vis_print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            modelG.module.save('latest')
            modelD.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
    else:
        if epoch % opt.save_epoch_freq == 0:
            visualizer.vis_print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
            modelG.module.save('latest')
            modelD.module.save('latest')
            modelG.module.save(epoch)
            modelD.module.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

def update_models(opt, epoch, modelG, modelD, data_loader):
    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        modelG.module.update_learning_rate(epoch, 'G')
        modelD.module.update_learning_rate(epoch, 'D')

    ### gradually grow training sequence length
    if (epoch % opt.niter_step) == 0:
        data_loader.dataset.update_training_batch(epoch//opt.niter_step)
        modelG.module.update_training_batch(epoch//opt.niter_step)

    ### finetune all scales
    if (opt.n_scales_spatial > 1) and (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        modelG.module.update_fixed_params()   