"""Local shim replacing the `nerv` dependency of the vendored SlotFormer SAVi code.

savi.py upstream imports four names from nerv (https://github.com/Wuziyi616/nerv,
MIT, see LICENSE.nerv; definitions copied at commit 5709625763c424a8b81b06c8cc2724d6454e688c):
`BaseModel`, `deconv_out_shape`, `conv_norm_act`, `deconv_norm_act`. Vendoring all of
nerv (a full training framework) for four names would violate the minimal-vendoring
policy (D5), so they are inlined here. `BaseModel` is reduced to a plain `nn.Module`
alias: its nerv extras are training-framework hooks our code never calls (we use
scjepa's own training loop). The conv helpers are copied verbatim minus unused options.
See PROVENANCE.md.
"""

import torch.nn as nn


class BaseModel(nn.Module):
    """Minimal stand-in for `nerv.training.BaseModel` (training hooks unused here)."""


def deconv_out_shape(in_size, stride, padding, kernel_size, out_padding, dilation=1):
    """Calculate the output shape of a ConvTranspose layer (nerv/models/utils.py)."""
    if isinstance(in_size, int):
        return (in_size - 1) * stride - 2 * padding + dilation * (
            kernel_size - 1) + out_padding + 1
    elif isinstance(in_size, (tuple, list)):
        return type(in_size)((deconv_out_shape(s, stride, padding, kernel_size,
                                               out_padding, dilation)
                              for s in in_size))
    else:
        raise TypeError(f'Got invalid type {type(in_size)} for `in_size`')


def _get_normalizer(norm, channels, groups=16):
    """Get normalization layer (nerv/models/modules.py, 2d only)."""
    if norm == '':
        return nn.Identity()
    elif norm == 'bn':
        return nn.BatchNorm2d(channels)
    elif norm == 'gn':
        return nn.GroupNorm(groups, channels)
    elif norm == 'in':
        return nn.InstanceNorm2d(channels)
    elif norm == 'ln':
        return nn.LayerNorm(channels)
    else:
        raise ValueError(f'Normalizer {norm} not supported!')


def _get_act_func(act):
    """Get activation function (nerv/models/modules.py, subset used by savi.py)."""
    if act == '':
        return nn.Identity()
    elif act == 'relu':
        return nn.ReLU()
    else:
        raise ValueError(f'Activation function {act} not supported!')


def conv_norm_act(in_channels, out_channels, kernel_size, stride=1, norm='bn',
                  act='relu'):
    """Conv - Norm - Act (nerv/models/modules.py, 2d only)."""
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=norm not in ['bn', 'in'],
    )
    return nn.Sequential(conv, _get_normalizer(norm, out_channels),
                         _get_act_func(act))


def deconv_norm_act(in_channels, out_channels, kernel_size, stride=1, norm='bn',
                    act='relu'):
    """ConvTranspose - Norm - Act (nerv/models/modules.py, 2d only)."""
    deconv = nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        output_padding=stride - 1,
        bias=norm not in ['bn', 'in'],
    )
    return nn.Sequential(deconv, _get_normalizer(norm, out_channels),
                         _get_act_func(act))
