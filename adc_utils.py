from torch import Tensor
import torch
from typing import Optional, List, Tuple, Union
from torch.nn import functional as F
import torch.ao.quantization.observer as observer
from torch.nn.modules.utils import _pair
from torch.ao.quantization.observer import MinMaxObserver, HistogramObserver
import math
import ast

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ObserverCls = MinMaxObserver #HistogramObserver

partial_sum_size=256
input_bit=8
weight_bit=6
output_bit=8
range_mode="minimum"
range_start=None
mode = "quantize"
truncate = False

def get_quant_values(input, observer, bitwidth, signedness='s'):
    """
    alpha, beta = torch.max(input), torch.min(input)
    s = (alpha - beta) / (2**bitwidth-1)
    zp = -1*torch.round(beta / s)
    min_ = 0
    max_ = (2**bitwidth)-1
    """
    #"""
    s = torch.max(torch.abs(input)) / (2**bitwidth - 1)
    zp = 0
    min_ = -2**(bitwidth-1)
    max_ = 2**(bitwidth-1)-1
    #"""
    return s, zp, min_, max_
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
    def forward(ctx, y, x_s, x_zp, w_s, w_zp, range_start, bitwidth):
        # Quantize
        y_q = y / (x_s * w_s)

        global truncate
        if truncate:
            if range_start is None or range_mode=="minimum":
                range_start_temp = torch.ceil(torch.log(torch.max(y_q))/torch.log(torch.Tensor([2.]).to(y_q.device)))
            if range_start is None:
                range_start = 0
            if range_mode=="minimum":
                range_start = max(range_start, range_start_temp)

            abs_clamp_min = 2**(range_start-1-(bitwidth-1))
            abs_clamp_max = 2**(range_start-1)-1
            y_q = torch.sign(y_q) * torch.clamp(torch.abs(y_q), abs_clamp_min, abs_clamp_max)

        # Dequantize
        y_fq = y_q * (x_s * w_s)
        #y_fq += torch.randint(2, (y_fq.shape[0],1,1,1)).to(y_fq.device) * 1e-3

        return y_fq

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None

class ADC_Conv2d(torch.nn.modules.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

        self.observers_initialized = False
        self.register_buffer('weight_org', self.weight.data.clone())
       
    def initialize_observers(self):
        self.observers_initialized = True
        self.input_observer = ObserverCls(quant_min=-2**(input_bit-1), quant_max=2**(input_bit-1)-1, qscheme=torch.per_tensor_symmetric, dtype=torch.qint32).to(device)
        self.output_observer = ObserverCls(quant_min=-2**(output_bit-1), quant_max=2**(output_bit-1)-1, qscheme=torch.per_tensor_symmetric, dtype=torch.qint32).to(device)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        input_, weight_ = input, weight
        
        global mode
        if mode == "quantize":
            if not self.observers_initialized:
                self.initialize_observers() 
            x_s, x_zp, w_s, w_zp = 1, 0, 1, 0
            input_.data, x_s, x_zp = FakeQuantize.apply(input_.data, self.input_observer, input_bit)
            weight_.data, w_s, w_zp = FakeQuantize.apply(self.weight_org, None, weight_bit)

        if mode == "observe":
            if not self.observers_initialized:
                self.initialize_observers() 
            self.input_observer.forward(input_)

        output = 0
        partial_sum_size = 1000
        group_size = int(input_.shape[1]/self.groups) #weight_.shape[1]
        for g_i in range(self.groups):
            input_group = input_[:,g_i*group_size:(g_i+1)*group_size,:,:]
            for c_i in range(math.ceil(input_group.shape[1]/partial_sum_size)):
                input_slice = input_group[:,c_i*partial_sum_size:(c_i+1)*partial_sum_size,:,:]
                weight_slice = weight_[:,c_i*partial_sum_size:(c_i+1)*partial_sum_size,:,:]
                if self.padding_mode != 'zeros':
                    output_slice = F.conv2d(F.pad(input_slice, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                    weight_slice, bias, self.stride,
                                    _pair(0), self.dilation, 1) #self.groups)
                else:
                    output_slice = F.conv2d(input_slice, weight_slice, bias, self.stride,
                                    self.padding, self.dilation, 1) #self.groups)
                if False: #mode == "quantize":
                    output_slice = FakeTruncate.apply(output_slice, x_s, x_zp, w_s, w_zp, range_start, output_bit)
                output += output_slice
        
        if mode == "observe":
            self.output_observer.forward(output)
        
        return output

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
