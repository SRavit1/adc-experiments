from torch import Tensor
import torch
from typing import Optional, List, Tuple, Union
from torch.nn import functional as F
import torch.ao.quantization.observer as observer
from torch.nn.modules.utils import _pair
from torch.ao.quantization.observer import MinMaxObserver, HistogramObserver
import numpy as np
import math
import ast
from multiprocessing import Pool
import yaml
from load_config import load_config

args = load_config()

clamping_range_start = args.clamping_range_start # May be modified by main.py
clamping_range_mode = args.clamping_range_mode
conv_mode = "float"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ObserverCls = MinMaxObserver #HistogramObserver
clamping = False

s, zp = 0.03, 0

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_quant_values_observer(input, observer, bitwidth):
    observer = None
    if observer is not None:
        min_, max_ = observer.quant_min, observer.quant_max
        s, zp = observer.calculate_qparams()
    else:
        min_, max_ = (-2**(bitwidth-1)), (2**(bitwidth-1))-1
        s, zp = torch.Tensor([s]).to(device), torch.Tensor([0.0]).to(device)
    return s, zp, min_, max_

def get_quant_values_no_observer(input, bitwidth):
    if args.quant_sign == "asymmetric":
        alpha, beta = torch.max(input), torch.min(input)
        s = (alpha - beta) / (2**bitwidth-1)
        zp = -1*torch.round(beta / s)
        min_ = 0
        max_ = (2**bitwidth)-1
    elif conv_mode == "symmetric":
        s = torch.max(torch.abs(input)) / (2**bitwidth - 1)
        zp = 0
        min_ = -2**(bitwidth-1)
        max_ = 2**(bitwidth-1)-1
    return s, zp, min_, max_

class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, zp, min, max):
        ctx.s = s
        x = (x - zp) / s
        x = torch.round(x)
        x = torch.clamp(x, min, max)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output/ctx.s, None, None, None, None

class Dequantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, zp, min, max):
        ctx.s = s
        x = (x * s) + zp
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output*ctx.s, None, None, None, None

class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, zp, min, max):
        return x
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
        if clamping:
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
            return super().forward(input)
        else:
            return input

class ADC_Conv2d(torch.nn.modules.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

        self.observers_initialized = False
        self.input_observer = None
        self.truncate_input_observer = None

        self.truncate_input_mean_avg_meter = AverageMeter()
        self.truncate_input_var_avg_meter = AverageMeter()

        self.gain = torch.Tensor([1.]).to(device) if clamping_range_mode=="per_model" else torch.nn.parameter.Parameter(torch.Tensor([1.]))
        self.bn = MyBatchNorm2d(kargs[1])
        self.relu = True

        self.register_buffer('weight_org', self.weight.data.clone())
       
    def initialize_observers(self):
        self.observers_initialized = True
        self.input_observer = ObserverCls(quant_min=-2**(args.act_bit-1), quant_max=2**(args.act_bit-1)-1, qscheme=torch.per_tensor_symmetric, dtype=torch.qint32).to(device)
        self.truncate_input_observer = ObserverCls(quant_min=-2**(args.act_bit-1), quant_max=2**(args.act_bit-1)-1, qscheme=torch.per_tensor_symmetric, dtype=torch.qint32).to(device)
    
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
        
    def calculate_qparams(self):
        # use self.input_observer, args.act_bit, args.weight_bit, args.quant_sign
        self.x_s, self.x_zp = s, zp
        self.x_min, self.x_max = -2**args.act_bit, 2**args.act_bit-1
        self.w_s, self.w_zp = s, zp
        self.w_min, self.w_max = -2**args.weight_bit, 2**args.weight_bit-1

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        global mode
        global truncate
        input_, weight_ = input, weight

        if conv_mode == "observe":
            if not self.observers_initialized:
                self.initialize_observers() 
            self.input_observer.forward(input_)

        if conv_mode == "observe" or conv_mode == "quantize":
            input_.data = FakeQuantize.apply(input_.data, self.x_s, self.x_zp, self.x_min, self.x_max)
            weight_.data = FakeQuantize.apply(weight_.data, self.w_s, self.w_zp, self.w_min, self.w_max)

            if args.cim_signed_type == "unipolar":
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
                        if conv_mode == "observe":
                            self.truncate_input_observer.forward(output_slice)
                            if truncate_observe_mode == "mean":
                                self.truncate_input_mean_avg_meter.update(float(torch.mean(output_slice)))
                            elif truncate_observe_mode == "var":
                                self.truncate_input_var_avg_meter.update((float(torch.mean(output_slice))-self.truncate_input_mean_avg_meter.avg)**2)
                        output_slice = FakeADC.apply(output_slice, self.gain, self.x_s, self.w_s, args.clamping_range_start, args.act_bit)
                        output += output_slice
                else: # Depthwise convolution (assuming group_size < 128)
                    if self.padding_mode != 'zeros':
                        output = F.conv2d(F.pad(input_, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight_, bias, self.stride,
                                _pair(0), self.dilation, self.groups)
                    else:
                        output = F.conv2d(input_, weight_, bias, self.stride,
                            self.padding, self.dilation, self.groups)
                    if conv_mode == "observe":
                        self.truncate_input_observer.forward(output)
                    output = FakeADC.apply(output, self.gain, self.x_s, self.w_s, args.clamping_range_start, args.act_bit)
                
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
        
        output = self.bn(output)
        if self.relu:
            output = torch.nn.functional.relu(output)
        return output

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
