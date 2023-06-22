# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Re-implementation of torch.nn Modules by deriving them from ModuleLight instead of nn.Module for better efficiency.
The layers have to be defined inside a function to be picklable.
Being picklable is required for distributed training the GHN.

"""


import torch
import torch.nn.functional as F
import numbers
import operator
from collections import OrderedDict
from typing import Union, Optional
from torch.nn.modules.conv import _pair
from torch.nn.common_types import _size_2_t
from itertools import chain, islice


def create_light_modules(ModuleEmpty, ModuleLight):

    class Sequential(ModuleLight):
        def __init__(self, *args):
            super().__init__()
            if len(args) == 1 and isinstance(args[0], OrderedDict):
                for key, module in args[0].items():
                    self.add_module(key, module)
            else:
                for idx, module in enumerate(args):
                    self.add_module(str(idx), module)
            return

        def forward(self, input):
            for module in self:
                input = module(input)
            return input

        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return self.__class__(OrderedDict(list(self._modules.items())[idx]))
            else:
                return self._get_item_by_idx(self._modules.values(), idx)

        def _get_item_by_idx(self, iterator, idx):
            """Get the idx-th item of the iterator"""
            size = len(self)
            idx = operator.index(idx)
            if not -size <= idx < size:
                raise IndexError('index {} is out of range'.format(idx))
            idx %= size
            return next(islice(iterator, idx, None))

        def __len__(self) -> int:
            return len(self._modules)

        def __dir__(self):
            keys = super().__dir__()
            keys = [key for key in keys if not key.isdigit()]
            return keys

        def __iter__(self):
            return iter(self._modules.values())

        def append(self, module):
            self.add_module(str(len(self)), module)
            return self

    class ModuleList(ModuleLight):

        def __init__(self, modules=None) -> None:
            super().__init__()
            if modules is not None:
                self += modules

        def _get_abs_string_index(self, idx):
            """Get the absolute index for the list of modules"""
            idx = operator.index(idx)
            if not (-len(self) <= idx < len(self)):
                raise IndexError('index {} is out of range'.format(idx))
            if idx < 0:
                idx += len(self)
            return str(idx)

        def __getitem__(self, idx: int):
            if isinstance(idx, slice):
                return self.__class__(list(self._modules.values())[idx])
            else:
                return self._modules[self._get_abs_string_index(idx)]

        def __len__(self) -> int:
            return len(self._modules)

        def __iter__(self):
            return iter(self._modules.values())

        def __iadd__(self, modules):
            return self.extend(modules)

        def __add__(self, other):
            combined = ModuleList()
            for i, module in enumerate(chain(self, other)):
                combined.add_module(str(i), module)
            return combined

        def __dir__(self):
            keys = super().__dir__()
            keys = [key for key in keys if not key.isdigit()]
            return keys

        def append(self, module):
            self.add_module(str(len(self)), module)
            return self

        def extend(self, modules):
            offset = len(self)
            for i, module in enumerate(modules):
                self.add_module(str(offset + i), module)
            return self

    class AvgPool2d(ModuleEmpty):

        def __init__(self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None, padding: _size_2_t = 0,
                     ceil_mode: bool = False, count_include_pad: bool = True,
                     divisor_override: Optional[int] = None) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.stride = stride if (stride is not None) else kernel_size
            self.padding = padding
            self.ceil_mode = ceil_mode
            self.count_include_pad = count_include_pad
            self.divisor_override = divisor_override

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            y = F.avg_pool2d(input, self.kernel_size, self.stride,
                             self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)
            return y

    class MaxPool2d(ModuleEmpty):

        def __init__(self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None,
                     padding: _size_2_t = 0, dilation: _size_2_t = 1,
                     return_indices: bool = False, ceil_mode: bool = False) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.stride = stride if (stride is not None) else kernel_size
            self.padding = padding
            self.dilation = dilation
            self.return_indices = return_indices
            self.ceil_mode = ceil_mode

        def forward(self, input: torch.Tensor):
            return F.max_pool2d(input, self.kernel_size, self.stride,
                                self.padding, self.dilation, ceil_mode=self.ceil_mode,
                                return_indices=self.return_indices)

    class AdaptiveAvgPool2d(ModuleEmpty):

        def __init__(self, output_size: _size_2_t) -> None:
            super().__init__()
            self.output_size = output_size

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.adaptive_avg_pool2d(input, self.output_size)

    class ReLU(ModuleEmpty):

        def __init__(self, inplace: bool = False):
            super().__init__()
            self.inplace = inplace

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.relu(input, inplace=self.inplace)

    class GELU(ModuleEmpty):
        def __init__(self, approximate: str = 'none') -> None:
            super().__init__()
            self.approximate = approximate

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.gelu(input, approximate=self.approximate)

    class Hardswish(ModuleEmpty):

        def __init__(self, inplace: bool = False) -> None:
            super().__init__()
            self.inplace = inplace

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.hardswish(input, self.inplace)

    class Identity(ModuleEmpty):
        def forward(self, input):
            return input

    class Dropout(ModuleEmpty):

        def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
            super().__init__()
            self.p = p
            self.inplace = inplace

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.dropout(input, self.p, self.training, self.inplace)

    class Conv2d(ModuleLight):

        def __init__(self, in_channels: int,
                     out_channels: int,
                     kernel_size: _size_2_t,
                     stride: _size_2_t = 1,
                     padding: Union[str, _size_2_t] = 0,
                     dilation: _size_2_t = 1,
                     groups: int = 1,
                     bias: bool = True,
                     padding_mode: str = 'zeros',  # TODO: refine this type
                     device=None,
                     dtype=torch.bool):

            super().__init__()

            self._parameters = OrderedDict()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = _pair(kernel_size)
            self.stride = _pair(stride)
            self.padding = padding if isinstance(padding, str) else _pair(padding)
            self.dilation = _pair(dilation)
            self.groups = groups
            self.padding_mode = padding_mode
            self.weight = [out_channels, in_channels // groups, *self.kernel_size]
            if bias:
                self.bias = [out_channels]
            else:
                self.bias = None

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

    class Linear(ModuleLight):

        def __init__(self, in_features: int, out_features: int, bias: bool = True,
                     device=None, dtype=torch.bool):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features

            self.weight = [out_features, in_features]
            if bias:
                self.bias = [out_features]
            else:
                self.bias = None

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.linear(input, self.weight, self.bias)

    class BatchNorm2d(ModuleLight):

        def __init__(self, num_features,
                     eps=1e-5,
                     momentum=0.1,
                     affine=True,
                     track_running_stats=False,
                     device=None,
                     dtype=None
                     ):
            super().__init__()
            self.num_features = num_features
            self.eps = eps
            self.momentum = momentum
            self.affine = affine
            self.track_running_stats = track_running_stats

            assert affine and not track_running_stats, 'assumed affine and that running stats is not updated'
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

            self.weight = [num_features]
            self.bias = [num_features]

        def forward(self, input: torch.Tensor) -> torch.Tensor:

            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training:
                bn_training = True
            else:
                bn_training = (self.running_mean is None) and (self.running_var is None)

            return F.batch_norm(
                input,
                self.running_mean
                if not self.training or self.track_running_stats
                else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )

    class LayerNorm(ModuleLight):

        def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True,
                     device=None, dtype=None) -> None:
            super().__init__()

            if isinstance(normalized_shape, numbers.Integral):
                normalized_shape = (normalized_shape,)  # type: ignore[assignment]

            self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            assert elementwise_affine
            self.weight = list(normalized_shape)
            self.bias = list(normalized_shape)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)

    types = locals()
    types.pop('ModuleEmpty')
    types.pop('ModuleLight')
    return types
