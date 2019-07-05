#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import argparse, os, sys, subprocess
import setproctitle, colorama
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import *

import models, losses, datasets
from utils import flow_utils, tools

# fp32 copy of parameters for update
global param_copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--total_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', '-b', type=int, default=8, help="Batch size")
    parser.add_argument('--train_n_batches', type=int, default = -1, help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
    parser.add_argument('--crop_size', type=int, nargs='+', default = [256, 256], help="Spatial dimension to crop training samples for training")
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--schedule_lr_frequency', type=int, default=0, help='in number of iterations (0 for no schedule)')
    parser.add_argument('--schedule_lr_fraction', type=float, default=10)
    parser.add_argument("--rgb_max", type=float, default = 255.)

    parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')
    parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

    parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
    parser.add_argument('--validation_n_batches', type=int, default=-1)
    parser.add_argument('--render_validation', action='store_true', help='run inference (save flows to file) and every validation_frequency epoch')

    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_size', type=int, nargs='+', default = [-1,-1], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
    parser.add_argument('--inference_batch_size', type=int, default=1)
    parser.add_argument('--inference_n_batches', type=int, default=-1)
    parser.add_argument('--save_flow', action='store_true', help='save predicted flows to file')

    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")

    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--skip_validation', action='store_true')

    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024., help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')

    tools.add_arguments_for_module(parser, losses, argument_for_class='loss', default='L1Loss')

    tools.add_arguments_for_module(parser, torch.optim, argument_for_class='optimizer', default='Adam', skip_params=['params'])
    
    tools.add_arguments_for_module(parser, datasets, argument_for_class='training_dataset', default='MpiSintelFinal', 
                                    skip_params=['is_cropped'],
                                    parameter_defaults={'root': './MPI-Sintel/flow/training'})
    
    tools.add_arguments_for_module(parser, datasets, argument_for_class='validation_dataset', default='MpiSintelClean', 
                                    skip_params=['is_cropped'],
                                    parameter_defaults={'root': './MPI-Sintel/flow/training',
                                                        'replicates': 1})
    
    tools.add_arguments_for_module(parser, datasets, argument_for_class='inference_dataset', default='MpiSintelClean', 
                                    skip_params=['is_cropped'],
                                    parameter_defaults={'root': './MPI-Sintel/flow/training',
                                                        'replicates': 1})

    main_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_dir)

    # Parse the official arguments
    with tools.TimerBlock("Parsing Arguments") as block:
        args = parser.parse_args()
        if args.number_gpus < 0 : args.number_gpus = torch.cuda.device_count()

        # Get argument defaults (hastag #thisisahack)
        parser.add_argument('--IGNORE',  action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))

        # Print all arguments, color the non-defaults
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

        args.model_class = tools.module_to_dict(models)[args.model]
        args.optimizer_class = tools.module_to_dict(torch.optim)[args.optimizer]
        args.loss_class = tools.module_to_dict(losses)[args.loss]

        args.training_dataset_class = tools.module_to_dict(datasets)[args.training_dataset]
        args.validation_dataset_class = tools.module_to_dict(datasets)[args.validation_dataset]
        args.inference_dataset_class = tools.module_to_dict(datasets)[args.inference_dataset]

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        args.current_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).rstrip()
        args.log_file = join(args.save, 'args.txt')

        # dict to collect activation gradients (for training debug purpose)
        args.grads = {}

        if args.inference:
            args.skip_validation = True
            args.skip_training = True
            args.total_epochs = 1
            args.inference_dir = "{}/inference".format(args.save)

    print('Source Code')
    print(('  Current Git Hash: {}\n'.format(args.current_hash)))

    # Change the title for `top` and `pkill` commands
    setproctitle.setproctitle(args.save)

    # Dynamically load the dataset class with parameters passed in via "--argument_[param]=[value]" arguments
    with tools.TimerBlock("Initializing Datasets") as block:
        args.effective_batch_size = args.batch_size * args.number_gpus
        args.effective_inference_batch_size = args.inference_batch_size * args.number_gpus
        args.effective_number_workers = args.number_workers * args.number_gpus
        gpuargs = {'num_workers': args.effective_number_workers, 
                   'pin_memory': True, 
                   'drop_last' : True} if args.cuda else {}
        inf_gpuargs = gpuargs.copy()
        inf_gpuargs['num_workers'] = args.number_workers

        if exists(args.training_dataset_root):
            train_dataset = args.training_dataset_class(args, True, **tools.kwargs_from_args(args, 'training_dataset'))
            block.log('Training Dataset: {}'.format(args.training_dataset))
            block.log('Training Input: {}'.format(' '.join([str([d for d in x.size()]) for x in train_dataset[0][0]])))
            block.log('Training Targets: {}'.format(' '.join([str([d for d in x.size()]) for x in train_dataset[0][1]])))
            train_loader = DataLoader(train_dataset, batch_size=args.effective_batch_size, shuffle=True, **gpuargs)

        if exists(args.validation_dataset_root):
            validation_dataset = args.validation_dataset_class(args, True, **tools.kwargs_from_args(args, 'validation_dataset'))
            block.log('Validation Dataset: {}'.format(args.validation_dataset))
            block.log('Validation Input: {}'.format(' '.join([str([d for d in x.size()]) for x in validation_dataset[0][0]])))
            block.log('Validation Targets: {}'.format(' '.join([str([d for d in x.size()]) for x in validation_dataset[0][1]])))
            validation_loader = DataLoader(validation_dataset, batch_size=args.effective_batch_size, shuffle=False, **gpuargs)

        if exists(args.inference_dataset_root):
            inference_dataset = args.inference_dataset_class(args, False, **tools.kwargs_from_args(args, 'inference_dataset'))
            block.log('Inference Dataset: {}'.format(args.inference_dataset))
            block.log('Inference Input: {}'.format(' '.join([str([d for d in x.size()]) for x in inference_dataset[0][0]])))
            block.log('Inference Targets: {}'.format(' '.join([str([d for d in x.size()]) for x in inference_dataset[0][1]])))
            inference_loader = DataLoader(inference_dataset, batch_size=args.effective_inference_batch_size, shuffle=False, **inf_gpuargs)

    # Dynamically load model and loss class with parameters passed in via "--model_[param]=[value]" or "--loss_[param]=[value]" arguments
    with tools.TimerBlock("Building {} model".format(args.model)) as block:
        class ModelAndLoss(nn.Module):
            def __init__(self, args):
                super(ModelAndLoss, self).__init__()
                kwargs = tools.kwargs_from_args(args, 'model')
                self.model = args.model_class(args, **kwargs)
                kwargs = tools.kwargs_from_args(args, 'loss')
                self.loss = args.loss_class(args, **kwargs)
                
            def forward(self, data, target, inference=False ):
                output = self.model(data)

                loss_values = self.loss(output, target)

                if not inference :
                    return loss_values
                else :
                    return loss_values, output

        model_and_loss = ModelAndLoss(args)

        block.log('Effective Batch Size: {}'.format(args.effective_batch_size))
        block.log('Number of parameters: {}'.format(sum([p.data.nelement() if p.requires_grad else 0 for p in model_and_loss.parameters()])))

        # assing to cuda or wrap with dataparallel, model and loss 
        if args.cuda and (args.number_gpus > 0) and args.fp16:
            block.log('Parallelizing')
            model_and_loss = nn.parallel.DataParallel(model_and_loss, device_ids=list(range(args.number_gpus)))

            block.log('Initializing CUDA')
            model_and_loss = model_and_loss.cuda().half()
            torch.cuda.manual_seed(args.seed) 
            param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model_and_loss.parameters()]

        elif args.cuda and args.number_gpus > 0:
            block.log('Initializing CUDA')
            model_and_loss = model_and_loss.cuda()
            block.log('Parallelizing')
            model_and_loss = nn.parallel.DataParallel(model_and_loss, device_ids=list(range(args.number_gpus)))
            torch.cuda.manual_seed(args.seed) 

        else:
            block.log('CUDA not being used')
            torch.manual_seed(args.seed)

        # Load weights if needed, otherwise randomly initialize
        if args.resume and os.path.isfile(args.resume):
            block.log("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if not args.inference:
                args.start_epoch = checkpoint['epoch']
            best_err = checkpoint['best_EPE']
            model_and_loss.module.model.load_state_dict(checkpoint['state_dict'])
            block.log("Loaded checkpoint '{}' (at epoch {})".format(args.resume, checkpoint['epoch']))

        elif args.resume and args.inference:
            block.log("No checkpoint found at '{}'".format(args.resume))
            quit()

        else:
            block.log("Random initialization")

        block.log("Initializing save directory: {}".format(args.save))
        if not os.path.exists(args.save):
            os.makedirs(args.save)

        train_logger = SummaryWriter(log_dir = os.path.join(args.save, 'train'), comment = 'training')
        validation_logger = SummaryWriter(log_dir = os.path.join(args.save, 'validation'), comment = 'validation')

    # Dynamically load the optimizer with parameters passed in via "--optimizer_[param]=[value]" arguments 
    with tools.TimerBlock("Initializing {} Optimizer".format(args.optimizer)) as block:
        kwargs = tools.kwargs_from_args(args, 'optimizer')
        if args.fp16:
            optimizer = args.optimizer_class([p for p in param_copy if p.requires_grad], **kwargs)
        else:
            optimizer = args.optimizer_class([p for p in model_and_loss.parameters() if p.requires_grad], **kwargs)
        for param, default in list(kwargs.items()):
            block.log("{} = {} ({})".format(param, default, type(default)))

    # Log all arguments to file
    for argument, value in sorted(vars(args).items()):
        block.log2file(args.log_file, '{}: {}'.format(argument, value))

    # Reusable function for training and validataion
    def train(args, epoch, start_iteration, data_loader, model, optimizer, logger, is_validate=False, offset=0):
        statistics = []
        total_loss = 0

        if is_validate:
            model.eval()
            title = 'Validating Epoch {}'.format(epoch)
            args.validation_n_batches = np.inf if args.validation_n_batches < 0 else args.validation_n_batches
            progress = tqdm(tools.IteratorTimer(data_loader), ncols=100, total=np.minimum(len(data_loader), args.validation_n_batches), leave=True, position=offset, desc=title)
        else:
            model.train()
            title = 'Training Epoch {}'.format(epoch)
            args.train_n_batches = np.inf if args.train_n_batches < 0 else args.train_n_batches
            progress = tqdm(tools.IteratorTimer(data_loader), ncols=120, total=np.minimum(len(data_loader), args.train_n_batches), smoothing=.9, miniters=1, leave=True, position=offset, desc=title)

        last_log_time = progress._time()
        for batch_idx, (data, target) in enumerate(progress):

            data, target = [Variable(d) for d in data], [Variable(t) for t in target]
            if args.cuda and args.number_gpus == 1:
                data, target = [d.cuda(non_blocking=True) for d in data], [t.cuda(non_blocking=True) for t in target]

            optimizer.zero_grad() if not is_validate else None
            losses = model(data[0], target[0])
            losses = [torch.mean(loss_value) for loss_value in losses] 
            loss_val = losses[0] # Collect first loss for weight update
            total_loss += loss_val.data[0]
            loss_values = [v.data[0] for v in losses]

            # gather loss_labels, direct return leads to recursion limit error as it looks for variables to gather'
            loss_labels = list(model.module.loss.loss_labels)

            assert not np.isnan(total_loss)

            if not is_validate and args.fp16:
                loss_val.backward()
                if args.gradient_clip:
                    torch.nn.utils.clip_grad_norm(model.parameters(), args.gradient_clip)

                params = list(model.parameters())
                for i in range(len(params)):
                   param_copy[i].grad = params[i].grad.clone().type_as(params[i]).detach()
                   param_copy[i].grad.mul_(1./args.loss_scale)
                optimizer.step()
                for i in range(len(params)):
                    params[i].data.copy_(param_copy[i].data)

            elif not is_validate:
                loss_val.backward()
                if args.gradient_clip:
                    torch.nn.utils.clip_grad_norm(model.parameters(), args.gradient_clip)
                optimizer.step()

            # Update hyperparameters if needed
            global_iteration = start_iteration + batch_idx
            if not is_validate:
                tools.update_hyperparameter_schedule(args, epoch, global_iteration, optimizer)
                loss_labels.append('lr')
                loss_values.append(optimizer.param_groups[0]['lr'])

            loss_labels.append('load')
            loss_values.append(progress.iterable.last_duration)

            # Print out statistics
            statistics.append(loss_values)
            title = '{} Epoch {}'.format('Validating' if is_validate else 'Training', epoch)

            progress.set_description(title + ' ' + tools.format_dictionary_of_losses(loss_labels, statistics[-1]))

            if ((((global_iteration + 1) % args.log_frequency) == 0 and not is_validate) or
                (is_validate and batch_idx == args.validation_n_batches - 1)):

                global_iteration = global_iteration if not is_validate else start_iteration

                logger.add_scalar('batch logs per second', len(statistics) / (progress._time() - last_log_time), global_iteration)
                last_log_time = progress._time()

                all_losses = np.array(statistics)

                for i, key in enumerate(loss_labels):
                    logger.add_scalar('average batch ' + str(key), all_losses[:, i].mean(), global_iteration)
                    logger.add_histogram(str(key), all_losses[:, i], global_iteration)

            # Reset Summary
            statistics = []

            if ( is_validate and ( batch_idx == args.validation_n_batches) ):
                break

            if ( (not is_validate) and (batch_idx == (args.train_n_batches)) ):
                break

        progress.close()

        return total_loss / float(batch_idx + 1), (batch_idx + 1)

    # Reusable function for inference
    def inference(args, epoch, data_loader, model, offset=0):

        model.eval()
        
        if args.save_flow or args.render_validation:
            flow_folder = "{}/inference/{}.epoch-{}-flow-field".format(args.save,args.name.replace('/', '.'),epoch)
            if not os.path.exists(flow_folder):
                os.makedirs(flow_folder)

        
        args.inference_n_batches = np.inf if args.inference_n_batches < 0 else args.inference_n_batches

        progress = tqdm(data_loader, ncols=100, total=np.minimum(len(data_loader), args.inference_n_batches), desc='Inferencing ', 
            leave=True, position=offset)

        statistics = []
        total_loss = 0
        for batch_idx, (data, target) in enumerate(progress):
            if args.cuda:
                data, target = [d.cuda(non_blocking=True) for d in data], [t.cuda(non_blocking=True) for t in target]
            data, target = [Variable(d) for d in data], [Variable(t) for t in target]

            # when ground-truth flows are not available for inference_dataset, 
            # the targets are set to all zeros. thus, losses are actually L1 or L2 norms of compute optical flows, 
            # depending on the type of loss norm passed in
            with torch.no_grad():
                losses, output = model(data[0], target[0], inference=True)

            losses = [torch.mean(loss_value) for loss_value in losses] 
            loss_val = losses[0] # Collect first loss for weight update
            total_loss += loss_val.data[0]
            loss_values = [v.data[0] for v in losses]

            # gather loss_labels, direct return leads to recursion limit error as it looks for variables to gather'
            loss_labels = list(model.module.loss.loss_labels)

            statistics.append(loss_values)
            # import IPython; IPython.embed()
            if args.save_flow or args.render_validation:
                for i in range(args.inference_batch_size):
                    _pflow = output[i].data.cpu().numpy().transpose(1, 2, 0)
                    flow_utils.writeFlow( join(flow_folder, '%06d.flo'%(batch_idx * args.inference_batch_size + i)),  _pflow)

            progress.set_description('Inference Averages for Epoch {}: '.format(epoch) + tools.format_dictionary_of_losses(loss_labels, np.array(statistics).mean(axis=0)))
            progress.update(1)

            if batch_idx == (args.inference_n_batches - 1):
                break

        progress.close()

        return

    # Primary epoch loop
    best_err = 1e8
    progress = tqdm(list(range(args.start_epoch, args.total_epochs + 1)), miniters=1, ncols=100, desc='Overall Progress', leave=True, position=0)
    offset = 1
    last_epoch_time = progress._time()
    global_iteration = 0

    for epoch in progress:
        if args.inference or (args.render_validation and ((epoch - 1) % args.validation_frequency) == 0):
            stats = inference(args=args, epoch=epoch - 1, data_loader=inference_loader, model=model_and_loss, offset=offset)
            offset += 1

        if not args.skip_validation and ((epoch - 1) % args.validation_frequency) == 0:
            validation_loss, _ = train(args=args, epoch=epoch - 1, start_iteration=global_iteration, data_loader=validation_loader, model=model_and_loss, optimizer=optimizer, logger=validation_logger, is_validate=True, offset=offset)
            offset += 1

            is_best = False
            if validation_loss < best_err:
                best_err = validation_loss
                is_best = True

            checkpoint_progress = tqdm(ncols=100, desc='Saving Checkpoint', position=offset)
            tools.save_checkpoint({   'arch' : args.model,
                                      'epoch': epoch,
                                      'state_dict': model_and_loss.module.model.state_dict(),
                                      'best_EPE': best_err}, 
                                      is_best, args.save, args.model)
            checkpoint_progress.update(1)
            checkpoint_progress.close()
            offset += 1

        if not args.skip_training:
            train_loss, iterations = train(args=args, epoch=epoch, start_iteration=global_iteration, data_loader=train_loader, model=model_and_loss, optimizer=optimizer, logger=train_logger, offset=offset)
            global_iteration += iterations
            offset += 1

            # save checkpoint after every validation_frequency number of epochs
            if ((epoch - 1) % args.validation_frequency) == 0:
                checkpoint_progress = tqdm(ncols=100, desc='Saving Checkpoint', position=offset)
                tools.save_checkpoint({   'arch' : args.model,
                                          'epoch': epoch,
                                          'state_dict': model_and_loss.module.model.state_dict(),
                                          'best_EPE': train_loss}, 
                                          False, args.save, args.model, filename = 'train-checkpoint.pth.tar')
                checkpoint_progress.update(1)
                checkpoint_progress.close()


        train_logger.add_scalar('seconds per epoch', progress._time() - last_epoch_time, epoch)
        last_epoch_time = progress._time()
    print("\n")
