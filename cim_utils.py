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

range_start = args.range_start # May be modified by main.py
range_mode = args.range_mode

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ObserverCls = MinMaxObserver #HistogramObserver

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

def get_quant_values(input, observer, bitwidth):
    observer = None
    if observer is not None:
        min_, max_ = observer.quant_min, observer.quant_max
        s, zp = observer.calculate_qparams()
    else:
        min_, max_ = (-2**(bitwidth-1)), (2**(bitwidth-1))-1
        s, zp = torch.Tensor([0.03]).to(device), torch.Tensor([0.0]).to(device)
    return s, zp, min_, max_

    # Approach 1
    """
    mode = "symmetric"
    if mode == "asymmetric":
        alpha, beta = torch.max(input), torch.min(input)
        s = (alpha - beta) / (2**bitwidth-1)
        zp = -1*torch.round(beta / s)
        min_ = 0
        max_ = (2**bitwidth)-1
    elif mode == "symmetric":
        s = torch.max(torch.abs(input)) / (2**bitwidth - 1)
        zp = 0
        min_ = -2**(bitwidth-1)
        max_ = 2**(bitwidth-1)-1
    """

    # Approach 2
    """
    if observer is not None:
        min_, max_ = observer.quant_min, observer.quant_max
        s, zp = observer.calculate_qparams()
    elif signedness=='s':
        min_, max_ = -2**(bitwidth-1), 2*(bitwidth-1)-1
        s = max(min_, max_) / (2**(bitwidth-1) - 1)
        zp = 0
    elif signedness=='u':
        min_, max_ = 0, (2**bitwidth)-1
        s = (max_-min_) / (2**bitwidth - 1)
        zp = -1*np.round(min_ / s)
    return s, zp, min_, max_
    """

"""
x- tensor to be quantized
observer- provides s/zp/input_min/input_max
bitwidth- bitwidth of quantization
"""
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, observer, bitwidth):
        s, zp, min_, max_ = get_quant_values(x, observer, bitwidth)
        
        # Quantize
        x = (x - zp) / s
        x = torch.round(x)
        x = torch.clamp(x, min_, max_)
        
        # Dequantize
        x = (x * s) + zp
        return x, s, zp

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class FakeTruncate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, y_mean, x_s, x_zp, w_s, w_zp, range_start, bitwidth):
        # Quantize
        y_q = y / (x_s * w_s)

        global truncate
        range_start_temp = range_start
        if range_start is None:
            range_start = output_bit - 1
        if args.range_mode=="per_layer":
            range_start = torch.ceil(torch.log(torch.Tensor([y_mean]))/torch.log(torch.Tensor([2.]))).to(y_q.device)
        if range_start_temp is not None:
            range_start = np.clip(range_start.cpu(), range_start_temp, range_start_temp+4).to(range_start.device)

        abs_clamp_min = 2**(range_start-(bitwidth-1))
        abs_clamp_max = 2**(range_start)-1
        y_q = torch.sign(y_q) * torch.clamp(torch.abs(y_q), abs_clamp_min, abs_clamp_max)

        # Dequantize
        y_fq = y_q * (x_s * w_s)

        return y_fq

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None

class ADC_Conv2d(torch.nn.modules.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

        self.observers_initialized = False
        self.input_observer = None
        self.truncate_input_observer = None

        self.truncate_input_mean_avg_meter = AverageMeter()
        self.truncate_input_var_avg_meter = AverageMeter()

        self.register_buffer('weight_org', self.weight.data.clone())
       
    def initialize_observers(self):
        self.observers_initialized = True
        self.input_observer = ObserverCls(quant_min=-2**(args.act_bit-1), quant_max=2**(args.act_bit-1)-1, qscheme=torch.per_tensor_symmetric, dtype=torch.qint32).to(device)
        self.truncate_input_observer = ObserverCls(quant_min=-2**(args.act_bit-1), quant_max=2**(args.act_bit-1)-1, qscheme=torch.per_tensor_symmetric, dtype=torch.qint32).to(device)
        
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        global mode
        global truncate
        input_, weight_ = input, weight

        if mode == "observe":
            if not self.observers_initialized:
                self.initialize_observers() 
            self.input_observer.forward(input_)

        if mode == "observe" or mode == "quantize":
            input_.data, x_s, x_zp = FakeQuantize.apply(input_.data, self.input_observer, args.act_bit)
            weight_.data, w_s, w_zp = FakeQuantize.apply(weight_.data, None, args.weight_bit) #self.weight_org

            if args.signed_weight_approach == "unipolar":
                loops = [(input_, weight_, 1.)]
            else:
                loops = [(input_, torch.relu(weight_), 1.), (input_, torch.relu(-weight_), -1.)]
            
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
                        if mode == "observe":
                            self.truncate_input_observer.forward(output_slice)
                            if truncate_observe_mode == "mean":
                                self.truncate_input_mean_avg_meter.update(float(torch.mean(output_slice)))
                            elif truncate_observe_mode == "var":
                                self.truncate_input_var_avg_meter.update((float(torch.mean(output_slice))-self.truncate_input_mean_avg_meter.avg)**2)
                        output_slice_mean = self.truncate_input_mean_avg_meter.avg if self.truncate_input_mean_avg_meter is not None else None
                        output_slice = FakeTruncate.apply(output_slice, output_slice_mean, x_s, x_zp, w_s, w_zp, range_start, args.act_bit)
                        output += output_slice
                else: # Depthwise convolution (assuming group_size < 128)
                    if self.padding_mode != 'zeros':
                        output = F.conv2d(F.pad(input_, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight_, bias, self.stride,
                                _pair(0), self.dilation, self.groups)
                    else:
                        output = F.conv2d(input_, weight_, bias, self.stride,
                            self.padding, self.dilation, self.groups)
                    if mode == "observe":
                        self.truncate_input_observer.forward(output)
                    output_mean = self.truncate_input_mean_avg_meter.avg if self.truncate_input_mean_avg_meter is not None else None
                    print(range_start)
                    output = FakeTruncate.apply(output, output_mean, x_s, x_zp, w_s, w_zp, range_start, args.act_bit)
                
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
        return self._conv_forward(input, self.weight, self.bias)
