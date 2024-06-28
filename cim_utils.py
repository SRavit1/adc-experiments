from torch import Tensor
import torch
from typing import Optional, List, Tuple, Union
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
import numpy as np
import math
import ast
from multiprocessing import Pool
import yaml
from load_config import load_config

args = None
conv_mode = None
clamping_range_start = None
clamping_range_mode = None
clamping = None
device = None
s = None
zp = None

def initialize_params(config_path):
    global args, conv_mode, clamping_range_start, clamping_range_mode, clamping, device, s, zp
    args = load_config(config_path)
    conv_mode = "float"
    clamping_range_start = args.clamping_range_start
    clamping_range_mode = args.clamping_range_mode
    clamping = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    s, zp = 0.03, 0

class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, zp, min, max):
        x = (x - zp) / s
        x = torch.round(x)
        x = torch.clamp(x, min, max)
        x = (x * s) + zp
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

class FakeADC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, gain, x_s, w_s, clamping_range_start, bitwidth):
        ctx.gain = gain
        ctx.y_sum = torch.sum(y)
        if clamping:
            y = y / (x_s * w_s)
        y = y * gain
        if clamping:
            y = y / (2**args.clamping_range_start)
            min, max = -(2**(bitwidth-1)), (2**(bitwidth-1))-1
            y = torch.round(torch.clamp(y, min, max))
            y = y * (2**args.clamping_range_start)
            y = y * (x_s * w_s)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output*ctx.gain, grad_output*ctx.y_sum, None, None, None, None

class MyBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.deactivated = False
    
    def forward(self, input):
        if not self.deactivated:
            print("BATCHNORM FORWARD")
            return super().forward(input)
        else:
            return input

class ADC_Conv2d(torch.nn.modules.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

        self.bn = MyBatchNorm2d(kargs[1])
        self.relu = True

        self.register_buffer('weight_org', self.weight.data.clone())
       
    def fuse_batchnorm(self):
        weight_mul = self.bn.weight/torch.sqrt(self.bn.running_var+1e-5)
        weight_mul = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(weight_mul, 1), 1), 1)
        bias_add = (-self.bn.running_mean * self.bn.weight / torch.sqrt(self.bn.running_var+1e-5)) + self.bn.bias
        self.weight.data *= weight_mul
        if self.bias is None:
            self.bias = torch.nn.parameter.Parameter(bias_add)
        else:
            self.bias.data += bias_add
        self.bn.deactivated = True
    
    def disable_batchnorm(self):
        self.bn.deactivated = True
        
    def calculate_qparams(self):
        self.x_s, self.x_zp = s, zp
        self.x_min, self.x_max = -2**args.act_bit, 2**args.act_bit-1
        self.w_s, self.w_zp = s, zp
        self.w_min, self.w_max = -2**args.weight_bit, 2**args.weight_bit-1

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        global mode
        global truncate
        input_, weight_, bias_ = input, weight, bias

        if conv_mode == "quantize" or conv_mode == "adc_quantize":
            input_.data = FakeQuantize.apply(input_.data, self.x_s, self.x_zp, self.x_min, self.x_max)
            weight_.data = FakeQuantize.apply(weight_.data, self.w_s, self.w_zp, self.w_min, self.w_max)
            if bias is not None:
                bias_.data = FakeQuantize.apply(bias_.data, self.w_s, self.w_zp, self.w_min, self.w_max)

            if conv_mode == "quantize" or args.cim_signed_type == "unipolar":
                loops = [(input_, weight_, 1.)]
            elif args.cim_signed_type == "differential_pair":
                loops = [(input_, torch.relu(weight_), 1.), (input_, torch.relu(-weight_), -1.)]
            else:
                raise Exception
            
            total_output = 0
            for input_, weight_, sign in loops:
                if self.groups == 1: # Standard convolution
                    output = 0
                    for c_i in range(math.ceil(input_.shape[1]/args.partial_sum_size)):
                        input_slice = input_[:,c_i*args.partial_sum_size:(c_i+1)*args.partial_sum_size,:,:]
                        weight_slice = weight_[:,c_i*args.partial_sum_size:(c_i+1)*args.partial_sum_size,:,:]
                        if self.padding_mode != 'zeros':
                            output_slice = F.conv2d(F.pad(input_slice, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                            weight_slice, bias, self.stride,
                                            _pair(0), self.dilation, 1) #self.groups)
                        else:
                            output_slice = F.conv2d(input_slice, weight_slice, bias, self.stride,
                                            self.padding, self.dilation, 1) #self.groups)
                        if conv_mode == "adc_quantize":
                            gain_val = self.gain if self.gain_enabled else 1
                            output_slice = FakeADC.apply(output_slice, gain_val, self.x_s, self.w_s, args.clamping_range_start, args.act_bit)
                        output += output_slice
                else: # Depthwise convolution (assuming group_size < 128)
                    if self.padding_mode != 'zeros':
                        output = F.conv2d(F.pad(input_, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight_, bias, self.stride,
                                _pair(0), self.dilation, self.groups)
                    else:
                        output = F.conv2d(input_, weight_, bias, self.stride,
                            self.padding, self.dilation, self.groups)
                    if conv_mode == "adc_quantize":
                        gain_val = self.gain if self.gain_enabled else 1
                        output = FakeADC.apply(output, gain_val, self.x_s, self.w_s, args.clamping_range_start, args.act_bit)
                
                output = output * sign
                total_output += output
            output = total_output
        else:
            if self.padding_mode != 'zeros':
                output = F.conv2d(F.pad(input_, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                          weight_, bias, self.stride,
                          _pair(0), self.dilation, self.groups)
            else:
                output = F.conv2d(input_, weight_, bias, self.stride,
                    self.padding, self.dilation, self.groups)
        return output

    def forward(self, input: Tensor) -> Tensor:
        out = self._conv_forward(input, self.weight, self.bias)
        out = self.bn(out)
        if self.relu:
            out = torch.nn.functional.relu(out)
        return out