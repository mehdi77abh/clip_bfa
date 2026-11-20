import torch
import bitsandbytes as bnb
from copy import deepcopy
import bitsandbytes as bnb

def find_all_bnbLinear(model,
    current_key_name=None,
    has_been_replaced=False,
    ):
    all_bnbLinear = set()
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        current_key_name_str = ".".join(current_key_name)
        if isinstance(module, bnb.nn.Linear8bitLt):
            all_bnbLinear.add(current_key_name_str)
            has_been_replaced = True
        elif isinstance(module, bnb.nn.Linear4bit):
            all_bnbLinear.add(current_key_name_str)
            has_been_replaced = True
        if len(list(module.children())) > 0:
            has_been_replaced, child_all_bnbLinear = find_all_bnbLinear(
                module,
                current_key_name,
                has_been_replaced=has_been_replaced,
            )
            all_bnbLinear |= child_all_bnbLinear
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return has_been_replaced, all_bnbLinear

def replace_with_myLinear(model,
    modules_to_convert=None,
    current_key_name=None,
    has_been_replaced=False,
    ):
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        current_key_name_str = ".".join(current_key_name)
        if current_key_name_str in modules_to_convert:
            src_cls = model._modules[name].source_cls
            tmp = model._modules[name]
            if isinstance(module, bnb.nn.Linear8bitLt):
                model._modules[name] = my_8bit_linear(tmp)
                has_been_replaced = True
            elif isinstance(module, bnb.nn.Linear4bit):
                model._modules[name] = my_4bit_linear(tmp)
                has_been_replaced = True
            # Store the module class in case we need to transpose the weight later
            model._modules[name].source_cls = src_cls
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_myLinear(
                module,
                modules_to_convert,
                current_key_name,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced

from functools import reduce  # Required in Python 3
import operator
import bitsandbytes.functional as F2
# math.prod not compatible with python < 3.8
def prod(iterable):
    return reduce(operator.mul, iterable, 1)

class my_8bit_linear(torch.nn.Module):
    def __init__(self, bnb_linear):
        super(my_8bit_linear, self).__init__()
        self.ori_bnb_linear = bnb_linear
        self.weight = self.ori_bnb_linear.weight
        self.state = self.ori_bnb_linear.state
        self.bias = self.ori_bnb_linear.bias
        self.device = self.ori_bnb_linear.weight.device
        
        self.ori_cb = self.ori_bnb_linear.weight.CB.clone().to(torch.float16).cuda()
        self.ori_shape = self.ori_cb.shape
        
        self.w_int = torch.nn.Parameter(self.ori_bnb_linear.weight.CB.clone().to(torch.float16), requires_grad=True)
        self.absmax = torch.nn.Parameter(self.weight.SCB.data.clone().to(torch.float16), requires_grad=True)

        self.ori_cb = torch.nn.Parameter(self.ori_bnb_linear.weight.CB.clone().to(torch.float16), requires_grad=True)
        self.is_train=False

    def forward(self, x: torch.Tensor):
        w = self.w_int.to(self.device)
        absmax = self.absmax.to(self.device)
        
        input_shape = x.shape
        shapeB = self.weight.shape
        if len(input_shape) == 3:
            output_shape = (input_shape[0], input_shape[1], shapeB[0])
        else:
            output_shape = (input_shape[0], shapeB[0])
        
        output = torch.nn.functional.linear(x, w)
        output = output.mul_(absmax.unsqueeze(0).mul(1.0 / 127.0))
        if self.bias is not None:
            output = output.add_(self.bias)
        
        return output.view(output_shape)

class my_4bit_linear(torch.nn.Module):
    compute_type_is_set = False
    def __init__(self, bnb_linear):
        super(my_4bit_linear, self).__init__()
        self.ori_bnb_linear = deepcopy(bnb_linear)
        self.weight = bnb_linear.weight
        self.bias = bnb_linear.bias
        self.quant_state = deepcopy(self.weight.quant_state)
        self.compute_dtype = torch.float32

        self.ori_absmax = self.quant_state.absmax
        self.blocksize = self.quant_state.state2.blocksize
        self.blocknum = self.ori_absmax.numel() // self.blocksize
        self.absmax_grad = None

        absmax = F2.dequantize_blockwise(self.ori_absmax, self.quant_state.state2)
        absmax += self.quant_state.offset
    
        self.real_weight = torch.nn.Parameter(F2.dequantize_4bit(self.weight.t(), self.quant_state).t(),
                                    requires_grad=True)
        self.ori_shape = self.quant_state.shape

    def forward(self, x: torch.Tensor):
        if prod(x.shape) == 0:
            B_shape = self.quant_state.shape
            if x.shape[-1] == B_shape[0]:
                return torch.empty(x.shape[:-1] + B_shape[1:], dtype=x.dtype, device=x.device)
            else:
                return torch.empty(x.shape[:-1] + B_shape[:1], dtype=x.dtype, device=x.device)

        real_weight = self.real_weight.to(x.dtype).cuda()
        
        output = torch.nn.functional.linear(x, real_weight, self.bias)
        return output