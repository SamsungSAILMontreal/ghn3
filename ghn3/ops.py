# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Layers and Networks factory.
Module base classes are dynamically constructed so that the same code can be used for both
training GHNs (lightweight modules are used) and training the baseline networks (standard nn.Modules are used).

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from .graphormer import create_transformer
from .light_ops import create_light_modules
from ppuda.deepnets1m.ops import parse_op_ks
from ppuda.deepnets1m.net import AuxiliaryHeadImageNet, AuxiliaryHeadCIFAR, drop_path, _is_none, named_layered_modules


def create_ops(light):

    if light:

        class ModuleEmpty:
            """
            Base class for layers without any trainable parameters.
            It avoids inheriting from nn.Module Network creation more efficient when training GHNs.
            """

            def __init__(self):
                self.training = True
                self._modules = {}
                self._named_modules = {}
                return

            def __reduce__(self):
                return _InitializeModule(), (self.__class__.__name__,), self.__dict__

            def to(self, *args, **kwargs):
                return None

            def named_modules(self):
                return self._named_modules

            def add_module(self, name, module):
                self.__dict__.get('_modules')[name] = module

            def __call__(self, *input, **kwargs):
                return self.forward(*input, **kwargs)

        class Module(ModuleEmpty):
            """
            Base class for layers with trainable parameters.
            It avoids inheriting from nn.Module Network creation more efficient when training GHNs.
            """

            def __init__(self):
                super().__init__()
                self._parameters = {}
                return

            def parameters(self, recurse: bool = True):

                for n, p in self._parameters.items():
                    if p is not None:
                        yield p

                for name, module in self._modules.items():
                    if not isinstance(module, Module):
                        continue
                    for p in module.parameters():
                        if p is not None:
                            yield p

            def named_modules(self, memo=None, prefix: str = '', remove_duplicate: bool = True):
                if memo is None:
                    memo = set()
                if self not in memo:
                    if remove_duplicate:
                        memo.add(self)
                    yield prefix, self
                    for name, module in self._modules.items():
                        if not isinstance(module, Module):
                            continue
                        submodule_prefix = prefix + ('.' if prefix else '') + name
                        for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                            yield m

            def __setattr__(self, name, value) -> None:
                if isinstance(value, (nn.Module, ModuleEmpty)):
                    self.__dict__.get('_modules')[name] = value
                elif isinstance(value, (torch.Tensor, nn.Parameter)) or \
                        (name in ['weight', 'bias'] and value is None) or \
                        isinstance(value, (list, tuple)) and name in ['weight', 'bias']:
                    self.__dict__.get('_parameters')[name] = value

                object.__setattr__(self, name, value)

        modules_light = create_light_modules(ModuleEmpty, Module)
        # for k in modules_light:
        #     exec(f'{k} = modules_light[k]')
        # the above is not working for some reason, so have to list all the modules explicitly

        Conv2d = modules_light['Conv2d']
        Linear = modules_light['Linear']
        Identity = modules_light['Identity']
        ReLU = modules_light['ReLU']
        GELU = modules_light['GELU']
        Hardswish = modules_light['Hardswish']
        Sequential = modules_light['Sequential']
        ModuleList = modules_light['ModuleList']
        Dropout = modules_light['Dropout']
        LayerNorm = modules_light['LayerNorm']
        BatchNorm2d = modules_light['BatchNorm2d']
        AvgPool2d = modules_light['AvgPool2d']
        MaxPool2d = modules_light['MaxPool2d']
        AdaptiveAvgPool2d = modules_light['AdaptiveAvgPool2d']

    else:

        from torch.nn.modules import Module, ModuleList, Sequential, Dropout, Identity, Linear, Conv2d, \
            BatchNorm2d, LayerNorm, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, ReLU, GELU, Hardswish
        ModuleEmpty = Module

    def bn_layer(norm, C):
        if norm in [None, '', 'none']:
            norm_layer = Identity()
        elif norm.startswith('bn'):
            norm_layer = BatchNorm2d(C, track_running_stats=norm.find('track') >= 0)
        else:
            raise NotImplementedError(norm)
        return norm_layer


    """
    Defining custom layers from https://github.com/facebookresearch/ppuda/blob/main/ppuda/deepnets1m/ops.py.
    But here we inherit them from our Module/ModuleEmpty instead of nn.Module for faster creation of module instances.
    """
    class Stride(ModuleEmpty):
        def __init__(self, stride):
            super().__init__()
            self.stride = stride

        def forward(self, x):
            if self.stride == 1:
                return x
            return x[:, :, ::self.stride, ::self.stride]

    class Zero(ModuleEmpty):
        def __init__(self, stride):
            super().__init__()
            self.stride = stride

        def forward(self, x):
            if self.stride == 1:
                return x.mul(0.)
            return x[:, :, ::self.stride, ::self.stride].mul(0.)

    class FactorizedReduce(Module):
        def __init__(self, C_in, C_out, norm='bn', stride=2):
            super().__init__()
            assert C_out % 2 == 0
            self.stride = stride
            self.relu = ReLU(inplace=False)
            self.conv_1 = Conv2d(C_in, C_out // 2, 1, stride=stride, padding=0, bias=False)
            self.conv_2 = Conv2d(C_in, C_out // 2, 1, stride=stride, padding=0, bias=False)
            self.bn = bn_layer(norm, C_out)

        def forward(self, x):
            x = self.relu(x)
            out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:] if self.stride > 1 else x)], dim=1)
            out = self.bn(out)
            return out

    class ReLUConvBN(Module):

        def __init__(self, C_in, C_out, ks=1, stride=1, padding=0, norm='bn', double=False):
            super().__init__()
            self.stride = stride
            if double:
                conv = [
                    Conv2d(C_in, C_in, (1, ks), stride=(1, stride), padding=(0, padding), bias=False),
                    Conv2d(C_in, C_out, (ks, 1), stride=(stride, 1), padding=(padding, 0), bias=False)]
            else:
                conv = [Conv2d(C_in, C_out, ks, stride=stride, padding=padding, bias=False)]
            self.op = Sequential(
                ReLU(inplace=False),
                *conv,
                bn_layer(norm, C_out))

        def forward(self, x):
            return self.op(x)

    class DilConv(Module):

        def __init__(self, C_in, C_out, ks, stride, padding, dilation, norm='bn'):
            super().__init__()
            self.stride = stride

            self.op = Sequential(
                ReLU(inplace=False),
                Conv2d(C_in, C_in, kernel_size=ks, stride=stride, padding=padding, dilation=dilation,
                       groups=C_in, bias=False),
                Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                bn_layer(norm, C_out)
            )

        def forward(self, x):
            return self.op(x)

    class SepConv(Module):

        def __init__(self, C_in, C_out, ks, stride, padding, norm='bn'):
            super().__init__()
            self.stride = stride

            self.op = Sequential(
                ReLU(inplace=False),
                Conv2d(C_in, C_in, kernel_size=ks, stride=stride, padding=padding, groups=C_in, bias=False),
                Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
                bn_layer(norm, C_in),
                ReLU(inplace=False),
                Conv2d(C_in, C_in, kernel_size=ks, stride=1, padding=padding, groups=C_in, bias=False),
                Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                bn_layer(norm, C_out)
            )

        def forward(self, x):
            return self.op(x)

    class ChannelSELayer(Module):
        def __init__(self, num_channels, reduction_ratio=2, dim_out=None, stride=1):
            """
            :param num_channels: No of input channels
            :param reduction_ratio: By how much should the num_channels should be reduced
            """
            super().__init__()

            if dim_out is not None:
                assert dim_out == num_channels, (dim_out, num_channels, 'only same dimensionality is supported')
            num_channels_reduced = num_channels // reduction_ratio
            self.reduction_ratio = reduction_ratio
            self.stride = stride
            self.fc1 = Linear(num_channels, num_channels_reduced, bias=True)
            self.fc2 = Linear(num_channels_reduced, num_channels, bias=True)
            self.relu = ReLU(inplace=True)
            self.sigmoid = Hardswish()

        def forward(self, input_tensor):
            """
            :param input_tensor: X, shape = (batch_size, num_channels, H, W)
            :return: output tensor
            """

            batch_size, num_channels, H, W = input_tensor.size()
            # Average along each channel
            squeeze_tensor = input_tensor.reshape(batch_size, num_channels, -1).mean(dim=2)

            # channel excitation
            fc_out_1 = self.fc1(squeeze_tensor)
            fc_out_2 = self.fc2(self.relu(fc_out_1))

            a, b = squeeze_tensor.size()
            output_tensor = torch.mul(input_tensor, self.sigmoid(fc_out_2).view(a, b, 1, 1))
            if self.stride > 1:
                output_tensor = output_tensor[:, :, ::self.stride, ::self.stride]
            return output_tensor

    transformer_types = create_transformer(Module, Linear, GELU, ReLU, LayerNorm, Dropout, Identity, Sequential)

    class PosEnc(Module):
        def __init__(self, C, ks):
            super().__init__()
            self.weight = [1, C, ks, ks] if light else nn.Parameter(torch.randn(1, C, ks, ks))

        def forward(self, x):
            """
            Args:
                x: Tensor, shape [batch_size, seq_len, embedding_dim]
            """
            try:
                return x + self.weight
            except:
                print(x.shape, self.weight.shape)
                raise

    OPS = {
        # i, o, k, s, n = C_in, C_out, ks, stride, norm
        'none': lambda i, o, k, s, n: Zero(s),
        'skip_connect': lambda i, o, k, s, n: Identity() if s == 1 else FactorizedReduce(i, o, norm=n),
        'avg_pool': lambda i, o, k, s, n: AvgPool2d(k, stride=s, padding=k // 2, count_include_pad=False),
        'max_pool': lambda i, o, k, s, n: MaxPool2d(k, stride=s, padding=k // 2),
        'conv': lambda i, o, k, s, n: ReLUConvBN(i, o, k, s, k // 2, norm=n),
        'sep_conv': lambda i, o, k, s, n: SepConv(i, o, k, s, k // 2, norm=n),
        'dil_conv': lambda i, o, k, s, n: DilConv(i, o, k, s, k - k % 2, 2, norm=n),
        'conv2': lambda i, o, k, s, n: ReLUConvBN(i, o, k, s, k // 2, norm=n, double=True),
        'conv_stride': lambda i, o, k, s, n: Conv2d(i, o, k, stride=k, bias=False, padding=int(k < 4)),
        'msa': lambda i, o, k, s, n: transformer_types['TransformerLayer'](i, stride=s),
        'cse': lambda i, o, k, s, n: ChannelSELayer(i, dim_out=o, stride=s),
    }

    class Cell(Module):

        def __init__(self, genotype, C_prev_prev, C_prev, C_in, C_out, reduction, reduction_prev,
                     norm='bn', preproc=True, is_vit=False, cell_ind=0):
            super(Cell, self).__init__()

            self._is_vit = is_vit
            self._cell_ind = cell_ind
            self._has_none = sum([n[0] == 'none' for n in genotype.normal + genotype.reduce]) > 0
            self.genotype = genotype

            if preproc:
                if reduction_prev and not is_vit:
                    self.preprocess0 = FactorizedReduce(C_prev_prev, C_out, norm=norm)
                else:
                    self.preprocess0 = ReLUConvBN(C_prev_prev, C_out, norm=norm)
                self.preprocess1 = ReLUConvBN(C_prev, C_out, norm=norm)
            else:
                if reduction_prev and not is_vit:
                    self.preprocess0 = Stride(stride=2)
                else:
                    self.preprocess0 = Identity()
                self.preprocess1 = Identity()

            if reduction:
                op_names, indices = zip(*genotype.reduce)
                concat = genotype.reduce_concat
            else:
                op_names, indices = zip(*genotype.normal)
                concat = genotype.normal_concat
            self._compile(C_in, C_out, op_names, indices, concat, reduction, norm)

        def _compile(self, C_in, C_out, op_names, indices, concat, reduction, norm):
            assert len(op_names) == len(indices)
            self._steps = len(op_names) // 2
            self._concat = concat
            self.multiplier = len(concat)

            self._ops = ModuleList()
            for i, (name, index) in enumerate(zip(op_names, indices)):
                stride = 2 if (reduction and index < 2 and not self._is_vit) else 1
                name, ks = parse_op_ks(name)
                self._ops.append(OPS[name](C_in if index <= 1 else C_out, C_out, ks, stride, norm))

            self._indices = indices

        def forward(self, s0, s1, drop_path_prob=0):

            s0 = None if (s0 is None or _is_none(self.preprocess0)) else self.preprocess0(s0)
            s1 = None if (s1 is None or _is_none(self.preprocess1)) else self.preprocess1(s1)

            states = [s0, s1]
            for i in range(self._steps):
                h1 = states[self._indices[2 * i]]
                h2 = states[self._indices[2 * i + 1]]
                op1 = self._ops[2 * i]
                op2 = self._ops[2 * i + 1]
                s = None

                if not (isinstance(op1, Zero) or _is_none(op1) or h1 is None):
                    h1 = op1(h1)
                    if self.training and drop_path_prob > 0 and not isinstance(op1, Identity):
                        h1 = drop_path(h1, drop_path_prob)
                    s = h1

                if not (isinstance(op2, Zero) or _is_none(op2) or h2 is None):
                    h2 = op2(h2)

                    if self.training and drop_path_prob > 0 and not isinstance(op2, Identity):
                        h2 = drop_path(h2, drop_path_prob)
                    try:
                        s = h2 if s is None else (h1 + h2)
                    except:
                        print(h1.shape, h2.shape, self.genotype)
                        raise

                states.append(s)

            if sum([states[i] is None for i in self._concat]) > 0:
                # Replace None states with Zeros to match feature dimensionalities and enable forward pass
                assert self._has_none, self.genotype
                s_dummy = None
                for i in self._concat:
                    if states[i] is not None:
                        s_dummy = states[i] * 0
                        break

                if s_dummy is None:
                    return None
                else:
                    for i in self._concat:
                        if states[i] is None:
                            states[i] = s_dummy

            y = torch.cat([states[i] for i in self._concat], dim=1)
            return y

    class Network(Module):

        def __init__(self,
                     C,
                     num_classes,
                     genotype,
                     n_cells,
                     ks=3,
                     is_imagenet_input=True,
                     stem_pool=False,
                     stem_type=0,
                     imagenet_stride=4,
                     is_vit=None,
                     norm='bn-track',
                     preproc=True,
                     C_mult=2,
                     fc_layers=0,
                     fc_dim=0,
                     glob_avg=True,
                     auxiliary=False,
                     ):
            super().__init__()

            self.genotype = genotype
            self._C = C
            self._auxiliary = auxiliary
            self.drop_path_prob = 0
            self.expected_input_sz = 224 if is_imagenet_input else 32

            self._is_vit = sum(
                [n[0] == 'msa' for n in genotype.normal + genotype.reduce]) > 0 if is_vit is None else is_vit

            steps = len(genotype.normal_concat)  # number of inputs to the concatenation operation
            if steps > 1 or C_mult > 1:
                assert preproc, 'preprocessing layers must be used in this case'

            self._stem_type = stem_type
            assert stem_type in [0, 1], ('either 0 (simple) or 1 (imagenet-style) stem must be chosen', stem_type)

            C_prev_prev = C_prev = C_curr = C

            # Define the stem
            if self._is_vit:
                # Visual Transformer stem
                self.stem0 = OPS['conv_stride'](3, C, 16 if is_imagenet_input else 3, None, None)
                self.pos_enc = PosEnc(C, 14 if is_imagenet_input else 11)

            elif stem_type == 0:
                # Simple stem
                C_stem = int(C * (3 if (preproc and not is_imagenet_input) else 1))

                self.stem = Sequential(
                    Conv2d(3, C_stem, ks, stride=imagenet_stride if is_imagenet_input else 1, padding=ks // 2,
                               bias=False),
                    bn_layer(norm, C_stem),
                    MaxPool2d(3, stride=2, padding=1, ceil_mode=False) if stem_pool else Identity(),
                )

                C_prev_prev = C_prev = C_stem

            else:
                # ImageNet-style stem
                self.stem0 = Sequential(
                    Conv2d(3, C // 2, kernel_size=ks, stride=2 if is_imagenet_input else 1,
                               padding=ks // 2, bias=False),
                    bn_layer(norm, C // 2),
                    ReLU(inplace=True),
                    Conv2d(C // 2, C, kernel_size=3, stride=2 if is_imagenet_input else 1, padding=1, bias=False),
                    bn_layer(norm, C)
                )

                self.stem1 = Sequential(
                    ReLU(inplace=True),
                    Conv2d(C, C, 3, stride=2, padding=1, bias=False),
                    bn_layer(norm, C)
                )

            self._n_cells = n_cells
            self.cells = ModuleList()

            is_reduction = lambda cell_ind: cell_ind in [n_cells // 3, 2 * n_cells // 3] and cell_ind > 0
            self._auxiliary_cell_ind = 2 * n_cells // 3

            reduction_prev = stem_type == 1
            for cell_ind in range(n_cells):
                if is_reduction(cell_ind):
                    C_curr *= C_mult
                    reduction = True
                else:
                    reduction = False

                reduction_next = is_reduction(cell_ind + 1)

                cell = Cell(genotype,
                            C_prev_prev,
                            C_prev,
                            C_in=C_curr if preproc else C_prev,
                            C_out=C_curr * (C_mult if reduction_next and steps == 1 and not preproc else 1),
                            reduction=reduction,
                            reduction_prev=reduction_prev,
                            norm=norm,
                            is_vit=self._is_vit,
                            preproc=preproc,
                            cell_ind=cell_ind)
                self.cells.append(cell)

                reduction_prev = reduction
                C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

                if auxiliary and cell_ind == self._auxiliary_cell_ind:
                    if is_imagenet_input:
                        self.auxiliary_head = AuxiliaryHeadImageNet(C_prev, num_classes, norm=norm)
                    else:
                        self.auxiliary_head = AuxiliaryHeadCIFAR(C_prev, num_classes, norm=norm,
                                                                 pool_sz=2 if (stem_type == 1 or stem_pool) else 5)

            self._glob_avg = glob_avg
            if glob_avg:
                self.global_pooling = AdaptiveAvgPool2d(1)
            else:
                if is_imagenet_input:
                    s = 7 if (stem_type == 1 or stem_pool) else 14
                else:
                    s = 4 if (stem_type == 1 or stem_pool) else 8
                C_prev *= s ** 2

            fc = [Linear(C_prev, fc_dim if fc_layers > 1 else num_classes)]
            for i in range(fc_layers - 1):
                assert fc_dim > 0, fc_dim
                fc.append(ReLU(inplace=True))
                fc.append(Dropout(p=0.5, inplace=False))
                fc.append(Linear(in_features=fc_dim, out_features=fc_dim if i < fc_layers - 2 else num_classes))
            self.classifier = Sequential(*fc)

            if light:
                self.__dict__['_layered_modules'] = named_layered_modules(self)

        def forward(self, input):

            if self._is_vit:
                s0 = self.stem0(input)
                s0 = s1 = self.pos_enc(s0)
            else:
                if self._stem_type == 1:
                    s0 = self.stem0(input)
                    s1 = None if _is_none(self.stem1) else self.stem1(s0)
                else:
                    s0 = s1 = self.stem(input)

            logits_aux = None
            for cell_ind, cell in enumerate(self.cells):
                s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
                if self._auxiliary and cell_ind == self._auxiliary_cell_ind and self.training:
                    logits_aux = self.auxiliary_head(F.adaptive_avg_pool2d(s1, 8)
                                                     if self._is_vit and self.expected_input_sz == 32
                                                     else s1)

            if s1 is None:
                raise ValueError('the network has invalid configuration: the output is None')

            out = self.global_pooling(s1) if self._glob_avg else s1
            autocast = torch.cpu.amp.autocast if str(out.device) == 'cpu' else torch.cuda.amp.autocast
            with autocast(enabled=False):
                out = out.float()
                logits = self.classifier(out.reshape(out.size(0), -1))

            return logits, logits_aux

    types = locals()
    types.update(transformer_types)
    del types['transformer_types'], types['OPS'], types['light'], types['bn_layer']
    if light:
        del types['modules_light']
    return types  # return the local types as a dictionary


types_light = create_ops(light=True)
types_torch_nn = create_ops(light=False)

TransformerLayer = types_torch_nn['TransformerLayer']   # used to create GHN-3 in nn.py
PosEnc = types_torch_nn['PosEnc']
Network = types_torch_nn['Network']                     # for evaluating GHNs
NetworkLight = types_light['Network']                   # for training GHNs


class _InitializeModule:
    """
    This class enables pickling of the Network class in the multiprocessing/DDP case.
    This class must be defined at the module level for pickling to work.
    """

    def __call__(self, class_name):
        obj = _InitializeModule()
        obj.__class__ = types_light[class_name]
        return obj
