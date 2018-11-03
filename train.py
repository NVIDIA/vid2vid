### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

def train():
    opt = TrainOptions().parse()
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1    
        opt.nThreads = 1

    ### initialize dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    if opt.dataset_mode == 'pose':
        print('#training frames = %d' % dataset_size)
    else:
        print('#training videos = %d' % dataset_size)

    ### initialize models
    modelG, modelD, flowNet = create_model(opt)
    visualizer = Visualizer(opt)

    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    ### if continue training, recover previous states
    if opt.continue_train:        
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))   
        if start_epoch > opt.niter:
            modelG.module.update_learning_rate(start_epoch-1)
            modelD.module.update_learning_rate(start_epoch-1)
        if (opt.n_scales_spatial > 1) and (opt.niter_fix_global != 0) and (start_epoch > opt.niter_fix_global):
            modelG.module.update_fixed_params()
        if start_epoch > opt.niter_step:
            data_loader.dataset.update_training_batch((start_epoch-1)//opt.niter_step)
            modelG.module.update_training_batch((start_epoch-1)//opt.niter_step)
    else:    
        start_epoch, epoch_iter = 1, 0

    ### set parameters    
    n_gpus = opt.n_gpus_gen // opt.batchSize             # number of gpus used for generator for each batch
    tG, tD = opt.n_frames_G, opt.n_frames_D
    tDB = tD * opt.output_nc        
    s_scales = opt.n_scales_spatial
    t_scales = opt.n_scales_temporal
    input_nc = 1 if opt.label_nc != 0 else opt.input_nc
    output_nc = opt.output_nc     

    opt.print_freq = lcm(opt.print_freq, opt.batchSize)
    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    total_steps = total_steps // opt.print_freq * opt.print_freq  

    ### real training starts here  
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()    
        for idx, data in enumerate(dataset, start=epoch_iter):        
            if total_steps % opt.print_freq == 0:
                iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == 0
                    
            _, n_frames_total, height, width = data['B'].size()  # n_frames_total = n_frames_load * n_loadings + tG - 1        
            n_frames_total = n_frames_total // opt.output_nc
            n_frames_load = opt.max_frames_per_gpu * n_gpus      # number of total frames loaded into GPU at a time for each batch
            n_frames_load = min(n_frames_load, n_frames_total - tG + 1)
            t_len = n_frames_load + tG - 1                       # number of loaded frames plus previous frames

            fake_B_last = None  # the last generated frame from previous training batch (which becomes input to the next batch)
            real_B_all, fake_B_all, flow_ref_all, conf_ref_all = None, None, None, None # all real/generated frames so far
            if opt.sparse_D:
                real_B_all, fake_B_all, flow_ref_all, conf_ref_all = [None]*t_scales, [None]*t_scales, [None]*t_scales, [None]*t_scales
            real_B_skipped, fake_B_skipped = [None]*t_scales, [None]*t_scales           # temporally subsampled frames
            flow_ref_skipped, conf_ref_skipped = [None]*t_scales, [None]*t_scales       # temporally subsampled flows

            for i in range(0, n_frames_total-t_len+1, n_frames_load):
                # 5D tensor: batchSize, # of frames, # of channels, height, width
                input_A = Variable(data['A'][:, i*input_nc:(i+t_len)*input_nc, ...]).view(-1, t_len, input_nc, height, width)
                input_B = Variable(data['B'][:, i*output_nc:(i+t_len)*output_nc, ...]).view(-1, t_len, output_nc, height, width)                
                inst_A = Variable(data['inst'][:, i:i+t_len, ...]).view(-1, t_len, 1, height, width) if len(data['inst'].size()) > 2 else None

                ###################################### Forward Pass ##########################
                ####### generator                  
                fake_B, fake_B_raw, flow, weight, real_A, real_Bp, fake_B_last = modelG(input_A, input_B, inst_A, fake_B_last)            
                            
                if i == 0:
                    fake_B_first = fake_B[0, 0]   # the first generated image in this sequence
                real_B_prev, real_B = real_Bp[:, :-1], real_Bp[:, 1:]   # the collection of previous and current real frames

                ####### discriminator            
                ### individual frame discriminator          
                flow_ref, conf_ref = flowNet(real_B, real_B_prev)  # reference flows and confidences
                fake_B_prev = real_B_prev[:, 0:1] if fake_B_last is None else fake_B_last[0][:, -1:]
                if fake_B.size()[1] > 1:
                    fake_B_prev = torch.cat([fake_B_prev, fake_B[:, :-1].detach()], dim=1)
               
                losses = modelD(0, reshape([real_B, fake_B, fake_B_raw, real_A, real_B_prev, fake_B_prev, flow, weight, flow_ref, conf_ref]))
                losses = [ torch.mean(x) if x is not None else 0 for x in losses ]
                loss_dict = dict(zip(modelD.module.loss_names, losses))

                ### temporal discriminator
                loss_dict_T = []
                # get skipped frames for each temporal scale
                if t_scales > 0:
                    if opt.sparse_D:          
                        real_B_all, real_B_skipped = get_skipped_frames_sparse(real_B_all, real_B, t_scales, tD, n_frames_load, i)
                        fake_B_all, fake_B_skipped = get_skipped_frames_sparse(fake_B_all, fake_B, t_scales, tD, n_frames_load, i)
                        flow_ref_all, flow_ref_skipped = get_skipped_frames_sparse(flow_ref_all, flow_ref, t_scales, tD, n_frames_load, i, is_flow=True)
                        conf_ref_all, conf_ref_skipped = get_skipped_frames_sparse(conf_ref_all, conf_ref, t_scales, tD, n_frames_load, i, is_flow=True)
                    else:
                        real_B_all, real_B_skipped = get_skipped_frames(real_B_all, real_B, t_scales, tD)
                        fake_B_all, fake_B_skipped = get_skipped_frames(fake_B_all, fake_B, t_scales, tD)                
                        flow_ref_all, conf_ref_all, flow_ref_skipped, conf_ref_skipped = get_skipped_flows(flowNet, 
                            flow_ref_all, conf_ref_all, real_B_skipped, flow_ref, conf_ref, t_scales, tD)                         

                # run discriminator for each temporal scale
                for s in range(t_scales):                
                    if real_B_skipped[s] is not None:                        
                        losses = modelD(s+1, [real_B_skipped[s], fake_B_skipped[s], flow_ref_skipped[s], conf_ref_skipped[s]])
                        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
                        loss_dict_T.append(dict(zip(modelD.module.loss_names_T, losses)))

                # collect losses
                loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
                loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_VGG']
                loss_G += loss_dict['G_Warp'] + loss_dict['F_Flow'] + loss_dict['F_Warp'] + loss_dict['W']
                if opt.add_face_disc:
                    loss_G += loss_dict['G_f_GAN'] + loss_dict['G_f_GAN_Feat'] 
                    loss_D += (loss_dict['D_f_fake'] + loss_dict['D_f_real']) * 0.5
                      
                # collect temporal losses
                loss_D_T = []           
                t_scales_act = min(t_scales, len(loss_dict_T))            
                for s in range(t_scales_act):
                    loss_G += loss_dict_T[s]['G_T_GAN'] + loss_dict_T[s]['G_T_GAN_Feat'] + loss_dict_T[s]['G_T_Warp']                
                    loss_D_T.append((loss_dict_T[s]['D_T_fake'] + loss_dict_T[s]['D_T_real']) * 0.5)

                ###################################### Backward Pass ################################# 
                optimizer_G = modelG.module.optimizer_G
                optimizer_D = modelD.module.optimizer_D  
                # update generator weights            
                optimizer_G.zero_grad()            
                loss_G.backward()        
                optimizer_G.step()

                # update discriminator weights
                # individual frame discriminator
                optimizer_D.zero_grad()
                loss_D.backward()            
                optimizer_D.step()
                # temporal discriminator
                for s in range(t_scales_act):
                    optimizer_D_T = getattr(modelD.module, 'optimizer_D_T'+str(s))                
                    optimizer_D_T.zero_grad()
                    loss_D_T[s].backward()            
                    optimizer_D_T.step()        

            if opt.debug:
                call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == 0:
                t = (time.time() - iter_start_time) / opt.print_freq
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                for s in range(len(loss_dict_T)):
                    errors.update({k+str(s): v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_T[s].items()})            
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            ### display output images
            if save_fake:                
                if opt.label_nc != 0:
                    input_image = util.tensor2label(real_A[0, -1], opt.label_nc)
                elif opt.dataset_mode == 'pose':
                    input_image = util.tensor2im(real_A[0, -1, :3])
                    if real_A.size()[2] == 6:
                        input_image2 = util.tensor2im(real_A[0, -1, 3:])
                        input_image[input_image2 != 0] = input_image2[input_image2 != 0]
                else:
                    c = 3 if opt.input_nc == 3 else 1
                    input_image = util.tensor2im(real_A[0, -1, :c], normalize=False)
                if opt.use_instance:
                    edges = util.tensor2im(real_A[0, -1, -1:,...], normalize=False)
                    input_image += edges[:,:,np.newaxis]
                
                if opt.add_face_disc:
                    ys, ye, xs, xe = modelD.module.get_face_region(real_A[0, -1:])
                    if ys is not None:
                        input_image[ys, xs:xe, :] = input_image[ye, xs:xe, :] = input_image[ys:ye, xs, :] = input_image[ys:ye, xe, :] = 255 

                visual_list = [('input_image', input_image),
                               ('fake_image', util.tensor2im(fake_B[0, -1])),
                               ('fake_first_image', util.tensor2im(fake_B_first)),
                               ('fake_raw_image', util.tensor2im(fake_B_raw[0, -1])),
                               ('real_image', util.tensor2im(real_B[0, -1])),                                                          
                               ('flow_ref', util.tensor2flow(flow_ref[0, -1])),
                               ('conf_ref', util.tensor2im(conf_ref[0, -1], normalize=False))]
                if flow is not None:
                    visual_list += [('flow', util.tensor2flow(flow[0, -1])),
                                    ('weight', util.tensor2im(weight[0, -1], normalize=False))]
                visuals = OrderedDict(visual_list)                          
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == 0:
                visualizer.vis_print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                modelG.module.save('latest')
                modelD.module.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter > dataset_size - opt.batchSize:
                epoch_iter = 0
                break
           
        # end of epoch 
        iter_end_time = time.time()
        visualizer.vis_print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            visualizer.vis_print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
            modelG.module.save('latest')
            modelD.module.save('latest')
            modelG.module.save(epoch)
            modelD.module.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            modelG.module.update_learning_rate(epoch)
            modelD.module.update_learning_rate(epoch)

        ### gradually grow training sequence length
        if (epoch % opt.niter_step) == 0:
            data_loader.dataset.update_training_batch(epoch//opt.niter_step)
            modelG.module.update_training_batch(epoch//opt.niter_step)

        ### finetune all scales
        if (opt.n_scales_spatial > 1) and (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            modelG.module.update_fixed_params()    

def reshape(tensors):
    if isinstance(tensors, list):
        return [reshape(tensor) for tensor in tensors]
    if tensors is None:
        return None
    _, _, ch, h, w = tensors.size()
    return tensors.contiguous().view(-1, ch, h, w)

# get temporally subsampled frames for real/fake sequences
def get_skipped_frames(B_all, B, t_scales, tD):
    B_all = torch.cat([B_all.detach(), B], dim=1) if B_all is not None else B
    B_skipped = [None] * t_scales
    for s in range(t_scales):
        tDs = tD ** s        # number of skipped frames between neighboring frames (e.g. 1, 3, 9, ...)
        span = tDs * (tD-1)  # number of frames the final triplet frames span before skipping (e.g., 2, 6, 18, ...)
        n_groups = min(B_all.size()[1] - span, B.size()[1])
        if n_groups > 0:
            for t in range(0, n_groups, tD):
                skip = B_all[:, (-span-t-1):-t:tDs].contiguous() if t != 0 else B_all[:, -span-1::tDs].contiguous()                
                B_skipped[s] = torch.cat([B_skipped[s], skip]) if B_skipped[s] is not None else skip             
    max_prev_frames = tD ** (t_scales-1) * (tD-1)
    if B_all.size()[1] > max_prev_frames:
        B_all = B_all[:, -max_prev_frames:]
    return B_all, B_skipped

# get temporally subsampled frames for flows
def get_skipped_flows(flowNet, flow_ref_all, conf_ref_all, real_B, flow_ref, conf_ref, t_scales, tD):  
    flow_ref_skipped, conf_ref_skipped = [None] * t_scales, [None] * t_scales  
    flow_ref_all, flow = get_skipped_frames(flow_ref_all, flow_ref, 1, tD)
    conf_ref_all, conf = get_skipped_frames(conf_ref_all, conf_ref, 1, tD)
    if flow[0] is not None:
        flow_ref_skipped[0], conf_ref_skipped[0] = flow[0][:,1:], conf[0][:,1:]

    for s in range(1, t_scales):        
        if real_B[s] is not None and real_B[s].size()[1] == tD:
            flow_ref_skipped[s], conf_ref_skipped[s] = flowNet(real_B[s][:,1:], real_B[s][:,:-1])
    return flow_ref_all, conf_ref_all, flow_ref_skipped, conf_ref_skipped

def get_skipped_frames_sparse(B_all, B, t_scales, tD, n_frames_load, i, is_flow=False):
    B_skipped = [None] * t_scales
    _, _, ch, h, w = B.size()
    for s in range(t_scales):
        t_len = B_all[s].size()[1] if B_all[s] is not None else 0
        if t_len > 0 and (t_len % tD) == 0:
            B_all[s] = B_all[s][:, (-tD+1):] # get rid of unnecessary past frames        

        if s == 0:
            B_all[0] = torch.cat([B_all[0].detach(), B], dim=1) if B_all[0] is not None else B
        else:
            tDs = tD ** s
            idx_start = 0 if i == 0 else tDs - ((i-1) % tDs + 1)            
            if idx_start < n_frames_load:
                tmp = B[:, idx_start::tDs].contiguous()
                B_all[s] = torch.cat([B_all[s].detach(), tmp], dim=1) if B_all[s] is not None else tmp

        t_len = B_all[s].size()[1] if B_all[s] is not None else 0
        if t_len >= tD:            
            B_all[s] = B_all[s][:, (t_len % tD):]
            B_skipped[s] = B_all[s].view(-1, tD, ch, h, w)
            if is_flow:
                B_skipped[s] = B_skipped[s][:, 1:]

    return B_all, B_skipped

if __name__ == "__main__":
   train()