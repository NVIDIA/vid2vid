import numpy as np
import torch
import sys
from .base_model import BaseModel

class FlowNet(BaseModel):
    def name(self):
        return 'FlowNet'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # flownet 2           
        from .flownet2_pytorch import models as flownet2_models
        from .flownet2_pytorch.utils import tools as flownet2_tools
        from .flownet2_pytorch.networks.resample2d_package.resample2d import Resample2d
        
        self.flowNet = flownet2_tools.module_to_dict(flownet2_models)['FlowNet2']().cuda(self.gpu_ids[0])        
        checkpoint = torch.load('models/flownet2_pytorch/FlowNet2_checkpoint.pth.tar')
        self.flowNet.load_state_dict(checkpoint['state_dict'])
        self.flowNet.eval() 
        self.resample = Resample2d()
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input_A, input_B):        
        with torch.no_grad():
            size = input_A.size()
            assert(len(size) == 4 or len(size) == 5)
            if len(size) == 5:
                b, n, c, h, w = size
                input_A = input_A.view(-1, c, h, w)
                input_B = input_B.view(-1, c, h, w)
                flow, conf = self.compute_flow_and_conf(input_A, input_B)
                return flow.view(b, n, 2, h, w), conf.view(b, n, 1, h, w)
            else:
                return self.compute_flow_and_conf(input_A, input_B)

    def compute_flow_and_conf(self, im1, im2):
        assert(im1.size()[1] == 3)
        assert(im1.size() == im2.size())        
        old_h, old_w = im1.size()[2], im1.size()[3]
        new_h, new_w = old_h//64*64, old_w//64*64
        if old_h != new_h:
            downsample = torch.nn.Upsample(size=(new_h, new_w), mode='bilinear')
            upsample = torch.nn.Upsample(size=(old_h, old_w), mode='bilinear')
            im1 = downsample(im1)
            im2 = downsample(im2)
        self.flowNet.cuda(im1.get_device())
        data1 = torch.cat([im1.unsqueeze(2), im2.unsqueeze(2)], dim=2)            
        flow1 = self.flowNet(data1)
        conf = (self.norm(im1 - self.resample(im2, flow1)) < 0.02).float()
        if old_h != new_h:
            flow1 = upsample(flow1) * old_h / new_h
            conf = upsample(conf)
        return flow1.detach(), conf.detach()

    def norm(self, t):
        return torch.sum(t*t, dim=1, keepdim=True)   
