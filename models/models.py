### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch.nn as nn
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
    if opt.isTrain and len(opt.gpu_ids):
        modelD.initialize(opt)
        flowNet.initialize(opt)        
        if opt.n_gpus_gen == len(opt.gpu_ids):
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)
            modelD = nn.DataParallel(modelD, device_ids=opt.gpu_ids)
            flowNet = nn.DataParallel(flowNet, device_ids=opt.gpu_ids)
        else:             
            if opt.batchSize == 1:
                gpu_split_id = opt.n_gpus_gen + 1
                modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[0:1])                
            else:
                gpu_split_id = opt.n_gpus_gen
                modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[:gpu_split_id])
            modelD = nn.DataParallel(modelD, device_ids=opt.gpu_ids[gpu_split_id:] + [opt.gpu_ids[0]])
            flowNet = nn.DataParallel(flowNet, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
        return [modelG, modelD, flowNet]
    else:
        return modelG