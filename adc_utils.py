from torch import Tensor
import torch
from typing import Optional, List, Tuple, Union
from torch.nn import functional as F
import torch.ao.quantization.observer as observer
from torch.nn.modules.utils import _pair
from torch.ao.quantization.observer import MinMaxObserver, HistogramObserver
import math

ObserverCls = MinMaxObserver #HistogramObserver

partial_sum_size=256
input_bit=8
weight_bit=8
acc_output_bit=24
output_bit=8
range_start = 12
range_delta = 2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""
x- tensor to be quantized
observer- provides s/zp
quant_bits- if observer None, number of bits to quantize to
clamp_bits- None -> no clamping, not None -> clamp in given signed range
clamp_start_range- where clamping should start from
clamp_start_delta- number of nearby bits in which to search for clamp_start_range
"""
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, observer, quant_bits, clamp_bits, clamp_start_range, clamp_start_delta):
        if observer is not None:
            scales, zero_points = observer.calculate_qparams()
            zero_points = 0.
            input_min, input_max = observer.quant_min, observer.quant_max
        else:        
            scales = 2*torch.max(torch.abs(x.max()), torch.abs(x.min()))/(2**quant_bits)
            zero_points = 0.
            input_min, input_max = -2**(input_bit-1), 2**(input_bit-1)-1
         
        # Quantize
        x = (x - zero_points) / scales
        x = torch.round(x)
        x = torch.clamp(x, input_min, input_max)

        # Clamp (assume unsigned number, since after ReLU)
        if not clamp_bits is None:
            if not clamp_start_delta == 0:
                min_error = float('inf')
                min_search_val = torch.min(clamp_start_range-clamp_start_delta, 0)
                max_search_val = torch.max(clamp_start_range+clamp_start_delta, clamp_bits-1)
                for clamp_start_range_i in range(min_search_val, max_search_val+1):
                    curr_error = 0
                    if curr_error < min_error:
                        clamp_start_range = clamp_start_range_i
                        min_error = curr_error
            x = torch.clamp(x, 2**(clamp_start_range+1-clamp_bits), 2**(clamp_start_range+1)-1)

        # Dequantize
        x = (x * scales) + zero_points

        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

class ADC_Conv2d(torch.nn.modules.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.input_bit = input_bit
        self.acc_output_bit = acc_output_bit
        self.output_bit = output_bit
        self.weight_bit = weight_bit
        self.range_start = range_start
        self.range_delta = range_delta
        self.mode = "quantize" #"observe", "quantize", "float"

        self.bn = torch.nn.BatchNorm2d(self.out_channels)
        self.relu = False
        
        self.observers_initialized = False
       
    def initialize_observers(self):
        self.observers_initialized = True
        self.input_observer = ObserverCls(quant_min=-2**(self.input_bit-1), quant_max=2**(self.input_bit-1)-1, qscheme=torch.per_tensor_symmetric, dtype=torch.qint8).to(device)
        self.output_observer = ObserverCls(quant_min=-2**(self.acc_output_bit-1), quant_max=2**(self.acc_output_bit-1)-1, qscheme=torch.per_tensor_symmetric, dtype=torch.qint32).to(device)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        input_, weight_ = input, weight

        if self.mode == "float":
            if self.padding_mode != 'zeros':
                output = F.conv2d(F.pad(input_, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight_, bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            else:
                output = F.conv2d(input_, weight_, bias, self.stride,
                                self.padding, self.dilation, self.groups)
            output = self.bn(output)
            if self.relu:
                output = torch.relu(output)
            return output

        if self.mode == "quantize":
            input_fq = FakeQuantize.apply(input, self.input_observer, self.input_bit, None, None, None)
            weight_fq = FakeQuantize.apply(weight, None, self.weight_bit, None, None, None)
            input_, weight_ = input_fq, weight_fq

        if self.mode == "observe":
            if not self.observers_initialized:
                self.initialize_observers() 
            self.input_observer.forward(input)

        input_pos = torch.relu(input_)
        input_neg = -torch.relu(-input_)
        weight_pos = torch.relu(weight_)
        weight_neg = torch.relu(-weight_)
        assert torch.allclose(weight_, weight_pos-weight_neg)
        conv_inputs = [(input_, weight_, 1.)]
        #conv_inputs = [(input_, weight_pos, 1.), (input_, weight_neg, -1.)]

        output = 0
        for input_, weight_, sign in conv_inputs:
            pass_output = 0
            #slice_num = math.ceil(partial_sum_size/(weight_.shape[2]*weight_.shape[3]))
            slice_num = weight.shape[1]
            for c in range(0, weight.shape[1], slice_num):
                input_slice = input_[:,c:c+slice_num,:,:]
                weight_slice = weight_[:,c:c+slice_num,:,:]

                input_slice_pos = input_slice[input_slice>=0]
                input_slice_neg = input_slice[input_slice<0]
                weight_slice_pos = weight_slice[weight_slice>=0]
                weight_slice_neg = weight_slice[weight_slice<0]

                if self.padding_mode != 'zeros':
                    output_slice = F.conv2d(F.pad(input_slice, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                    weight_slice, bias, self.stride,
                                    _pair(0), self.dilation, self.groups)
                else:
                    output_slice = F.conv2d(input_slice, weight_slice, bias, self.stride,
                                    self.padding, self.dilation, self.groups)

                pass_output += output_slice

            pass_output *= sign
            output += pass_output
        
        pass_output = self.bn(pass_output)
        if self.relu:
            pass_output = torch.relu(pass_output)
        
        if self.mode == "observe":
            self.output_observer.forward(output)
        
        res = output
        return res

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
