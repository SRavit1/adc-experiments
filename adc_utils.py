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
weight_bit=8
acc_output_bit=24
output_bit=8
range_start=12
truncate=False

mode = "quantize"

def get_quant_values(observer, bitwidth, signedness='s'):
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
        s, zp, min_, max_ = get_quant_values(observer, bitwidth)
         
        # Quantize
        x = (x - zp) / s
        x = torch.round(x)
        x = torch.clamp(x, min_, max_)

        # Dequantize
        x = (x * s) + zp
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class FakeTruncate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, x_observer, x_bw, w_observer, w_bw, range_start):
        x_s, x_zp, x_min_, x_max_ = get_quant_values(x_observer, x_bw)
        w_s, w_zp, w_min_, w_max_ = get_quant_values(w_observer, w_bw)
        
        # Assuming x_zp_ and w_zp_ are both equal to 0

        # Quantize
        y_q = y / (x_s * w_s)

        global truncate
        if truncate:
            # Truncate
            if range_start is None: #per-layer quantization
                range_start = torch.ceil(torch.log(torch.max(y_q), 2.))
            y_q_t = torch.sign(y_q) * torch.clamp(torch.abs(y_q), 2**(range_start-(output_bit-1)), (2**range_start)-1)

        # Dequantize
        y_q_t_dq = y_q_t * (x_s * w_s)

        return y_q_t_dq

    @staticmethod
    def backward(ctx, grad_output):
        return 1, None, None, None, None, None

class ADC_Conv2d(torch.nn.modules.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

        self.observers_initialized = False
        self.register_buffer('weight_org', self.weight.data.clone())
       
    def initialize_observers(self):
        self.observers_initialized = True
        self.input_observer = ObserverCls(quant_min=-2**(input_bit-1), quant_max=2**(input_bit-1)-1, qscheme=torch.per_tensor_symmetric, dtype=torch.qint32).to(device)
        self.output_observer = ObserverCls(quant_min=-2**(acc_output_bit-1), quant_max=2**(acc_output_bit-1)-1, qscheme=torch.per_tensor_symmetric, dtype=torch.qint32).to(device)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        input_, weight_ = input, weight
        
        global mode
        if mode == "quantize":
            if not self.observers_initialized:
                self.initialize_observers() 
            #input_.data = FakeQuantize.apply(input_.data, self.input_observer, input_bit)
            #weight_.data = FakeQuantize.apply(self.weight_org, None, weight_bit)

        if mode == "observe":
            if not self.observers_initialized:
                self.initialize_observers() 
            self.input_observer.forward(input_)

        if self.padding_mode != 'zeros':
            output = F.conv2d(F.pad(input_, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight_, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        else:
            output = F.conv2d(input_, weight_, bias, self.stride,
                            self.padding, self.dilation, self.groups)
        
        if mode == "observe":
            self.output_observer.forward(output)
        
        return output

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
