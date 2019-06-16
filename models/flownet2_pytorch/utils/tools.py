# freda (todo) : 

import os, time, sys, math
import subprocess, shutil
from os.path import *
import numpy as np
from inspect import isclass
from pytz import timezone
from datetime import datetime
import inspect
import torch

def datestr():
    pacific = timezone('US/Pacific')
    now = datetime.now(pacific)
    return '{}{:02}{:02}_{:02}{:02}'.format(now.year, now.month, now.day, now.hour, now.minute)

def module_to_dict(module, exclude=[]):
        return dict([(x, getattr(module, x)) for x in dir(module)
                     if isclass(getattr(module, x))
                     and x not in exclude
                     and getattr(module, x) not in exclude])

class TimerBlock: 
    def __init__(self, title):
        print(("{}".format(title)))

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.clock()
        self.interval = self.end - self.start

        if exc_type is not None:
            self.log("Operation failed\n")
        else:
            self.log("Operation finished\n")


    def log(self, string):
        duration = time.clock() - self.start
        units = 's'
        if duration > 60:
            duration = duration / 60.
            units = 'm'
        print(("  [{:.3f}{}] {}".format(duration, units, string)))
    
    def log2file(self, fid, string):
        fid = open(fid, 'a')
        fid.write("%s\n"%(string))
        fid.close()

def add_arguments_for_module(parser, module, argument_for_class, default, skip_params=[], parameter_defaults={}):
    argument_group = parser.add_argument_group(argument_for_class.capitalize())

    module_dict = module_to_dict(module)
    argument_group.add_argument('--' + argument_for_class, type=str, default=default, choices=list(module_dict.keys()))
    
    args, unknown_args = parser.parse_known_args()
    class_obj = module_dict[vars(args)[argument_for_class]]

    argspec = inspect.getargspec(class_obj.__init__)

    defaults = argspec.defaults[::-1] if argspec.defaults else None

    args = argspec.args[::-1]
    for i, arg in enumerate(args):
        cmd_arg = '{}_{}'.format(argument_for_class, arg)
        if arg not in skip_params + ['self', 'args']:
            if arg in list(parameter_defaults.keys()):
                argument_group.add_argument('--{}'.format(cmd_arg), type=type(parameter_defaults[arg]), default=parameter_defaults[arg])
            elif (defaults is not None and i < len(defaults)):
                argument_group.add_argument('--{}'.format(cmd_arg), type=type(defaults[i]), default=defaults[i])
            else:
                print(("[Warning]: non-default argument '{}' detected on class '{}'. This argument cannot be modified via the command line"
                        .format(arg, module.__class__.__name__)))
            # We don't have a good way of dealing with inferring the type of the argument
            # TODO: try creating a custom action and using ast's infer type?
            # else:
            #     argument_group.add_argument('--{}'.format(cmd_arg), required=True)

def kwargs_from_args(args, argument_for_class):
    argument_for_class = argument_for_class + '_'
    return {key[len(argument_for_class):]: value for key, value in list(vars(args).items()) if argument_for_class in key and key != argument_for_class + 'class'}

def format_dictionary_of_losses(labels, values):
    try:
        string = ', '.join([('{}: {:' + ('.3f' if value >= 0.001 else '.1e') +'}').format(name, value) for name, value in zip(labels, values)])
    except (TypeError, ValueError) as e:
        print((list(zip(labels, values))))
        string = '[Log Error] ' + str(e)

    return string


class IteratorTimer():
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = self.iterable.__iter__()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        start = time.time()
        n = next(self.iterator)
        self.last_duration = (time.time() - start)
        return n

    next = __next__

def gpumemusage():
    gpu_mem = subprocess.check_output("nvidia-smi | grep MiB | cut -f 3 -d '|'", shell=True).replace(' ', '').replace('\n', '').replace('i', '')
    all_stat = [float(a) for a in gpu_mem.replace('/','').split('MB')[:-1]]

    gpu_mem = ''
    for i in range(len(all_stat)/2):
        curr, tot = all_stat[2*i], all_stat[2*i+1]
        util = "%1.2f"%(100*curr/tot)+'%'
        cmem = str(int(math.ceil(curr/1024.)))+'GB'
        gmem = str(int(math.ceil(tot/1024.)))+'GB'
        gpu_mem += util + '--' + join(cmem, gmem) + ' '
    return gpu_mem


def update_hyperparameter_schedule(args, epoch, global_iteration, optimizer):
    if args.schedule_lr_frequency > 0:
        for param_group in optimizer.param_groups:
            if (global_iteration + 1) % args.schedule_lr_frequency == 0:
                param_group['lr'] /= float(args.schedule_lr_fraction)
                param_group['lr'] = float(np.maximum(param_group['lr'], 0.000001))

def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')

