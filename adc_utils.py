from torch import Tensor
import torch
from typing import Optional, List, Tuple, Union
from torch.nn import functional as F
import torch.ao.quantization.observer as observer
from torch.nn.modules.utils import _pair
from torch.ao.quantization.observer import MinMaxObserver, HistogramObserver
import math

ObserverCls = MinMaxObserver

partial_sum_size=256
input_bit=8
weight_bit=6
acc_output_bit=24
output_bit=8

positive_weights=False #True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ThresholdSiLU(torch.nn.SiLU):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.input_threshold = torch.nn.Parameter(torch.tensor(0.))
    
    def forward(self, input_):
        # TODO: Change back to original
        return torch.nn.functional.relu(input_)
        #return super().forward(input_-self.input_threshold)

class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, observer, input_bit):
        if observer is not None:
            #observer = ObserverCls(quant_min=-2**(input_bit-1), quant_max=2**(input_bit-1)-1, qscheme=torch.per_tensor_symmetric).to(device)
            #observer.forward(x)
            scales, zero_points = observer.calculate_qparams()
            zero_points = 0
            input_min, input_max = observer.quant_min, observer.quant_max
        else:        
            #scales = (x.max()-x.min())/(2**input_bit)
            #zero_points = (x.max()+x.min())/2
            scales = 2*torch.max(torch.abs(x.max()), torch.abs(x.min()))/(2**input_bit)
            zero_points = 0
            input_min, input_max = -2**(input_bit-1), 2**(input_bit-1)-1
        

        x_q = x
        x_q = (x_q - zero_points) / scales
        #x_q = (x_q/scales) - zero_points
        x_q = torch.round(x_q)
        x_q = torch.clamp(x_q, input_min, input_max)

        #x_q = (x_q * scales) + zero_points

        ctx.scales = scales

        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output/ctx.scales, None, None, None

class ADC_Dequantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_observer, w, y, output_bit, best_range_start):
        #observer = ObserverCls(quant_min=-2**(input_bit-1), quant_max=2**(input_bit-1)-1, qscheme=torch.per_tensor_symmetric).to(device)
        #observer.forward(x)
        x_scales, x_zero_points = x_observer.calculate_qparams()
        x_zero_points = 0
             
        #scales = (x.max()-x.min())/(2**input_bit)
        #zero_points = (x.max()+x.min())/2
        w_scales = 2*torch.max(torch.abs(w.max()), torch.abs(w.min()))/(2**input_bit)
        w_zero_points = 0

        output_adc_q = y
            
        # Remove higher bits with modulus
        output_adc_q = torch.sign(output_adc_q) * (torch.abs(output_adc_q) % (2 ** (best_range_start)))
        
        # Remove lower bits with division
        val = (2 ** (best_range_start-output_bit+1))
        output_adc_q = output_adc_q // val
        output_adc_q = output_adc_q * val

        # Dequantize
        y_scales = x_scales * w_scales
        output_adc_q = output_adc_q * y_scales
        #x_q = (x_q+zero_points) * scales

        y = output_adc_q

        ctx.y_scales = y_scales

        return y

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, grad_output*ctx.y_scales, None, None

class ReLU_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.nn.functional.relu(x)
    def backward(ctx, grad_output):
        return grad_output

class ADC_Conv2d(torch.nn.modules.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.input_bit = input_bit
        self.acc_output_bit = acc_output_bit
        self.output_bit = output_bit
        self.weight_bit = weight_bit
        self.best_range_start = 0
        self.mode = "quantize" #"observe", "quantize", "float"

        self.observers_initialized = False
        
    def prepare_for_quantized_training(self):
        self.weight_float = torch.nn.parameter.Parameter(torch.zeros_like(self.weight))

    def initialize_observers(self):
        self.observers_initialized = True
        self.input_observer = ObserverCls(quant_min=-2**(self.input_bit-1), quant_max=2**(self.input_bit-1)-1, qscheme=torch.per_tensor_symmetric).to(device)
        #self.output_observer = ObserverCls(quant_min=-2**(self.acc_output_bit-1), quant_max=2**(self.acc_output_bit-1)-1, qscheme=torch.per_tensor_symmetric).to(device)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # if positive_weights:
        #     weight = ReLU_STE.apply(weight)
        # weight = torch.abs(weight)
        
        if self.mode == "observe":
            if not self.observers_initialized:
                self.initialize_observers()
            
            self.input_observer.forward(input)
            if self.padding_mode != 'zeros':
                output = F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight, bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            else:
                output = F.conv2d(input, weight, bias, self.stride,
                                self.padding, self.dilation, self.groups)
            
            #self.output_observer.forward(output)
            res = output
        elif self.mode == "quantize":
            # x, observer, input_bit
            input_q = Quantize.apply(input, self.input_observer, self.input_bit)
            weight_q = Quantize.apply(weight, None, self.weight_bit)
            # bias not used, so not quantizing

            slice_num = math.ceil(partial_sum_size/(weight.shape[2]*weight.shape[3]))
            output_adq = 0
            for c in range(0, weight.shape[1], slice_num):
                input_q_slice = input_q[:,c:c+slice_num,:,:]
                weight_q_slice = weight_q[:,c:c+slice_num,:,:]
                if self.padding_mode != 'zeros':
                    output_q_slice = F.conv2d(F.pad(input_q_slice, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                    weight_q_slice, bias, self.stride,
                                    _pair(0), self.dilation, self.groups)
                else:
                    output_q_slice = F.conv2d(input_q_slice, weight_q_slice, bias, self.stride,
                                    self.padding, self.dilation, self.groups)
                
                output_slice_adq = ADC_Dequantize.apply(self.input_observer, weight, output_q_slice, self.output_bit, self.best_range_start)
                output_adq += output_slice_adq
            
            res = output_adq
        elif self.mode == "float":
            if self.padding_mode != 'zeros':
                output = F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight, bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            else:
                output = F.conv2d(input, weight, bias, self.stride,
                                self.padding, self.dilation, self.groups)
            res = output
        return res

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
