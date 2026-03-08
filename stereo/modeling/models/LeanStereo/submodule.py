from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def disparity_regression(x, maxdisp):
    # assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea

    volume = volume.contiguous()
    return volume

class concat_volume_builder(nn.Module):
    def __init__(self, builder_type, maxdisp):
        super(concat_volume_builder, self).__init__()
        self.maxdisp = maxdisp
        self.builder_type = builder_type

        if self.builder_type == 'original':
            self.forward = self.build_concat_volume_original
        elif self.builder_type == 'reset':
            self.forward = self.build_concat_volume_reset
        elif self.builder_type == 'pad':
            self.forward = self.build_concat_volume_pad
        elif self.builder_type == 'clear':
            self.forward = self.build_concat_volume_clear
        else:
            raise ValueError("Unknown builder type {}".format(self.builder_type))
        print("Using concat volume builder type: {}".format(self.builder_type))
                

    # def forward(self, refimg_fea, targetimg_fea):
    #     return build_concat_volume(refimg_fea, targetimg_fea, self.maxdisp)
    def build_concat_volume_original(self,refimg_fea, targetimg_fea):
        maxdisp = self.maxdisp
        B, C, H, W = refimg_fea.shape
        volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
        for i in range(maxdisp):
            if i > 0:
                volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
                volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                volume[:, :C, i, :, :] = refimg_fea
                volume[:, C:, i, :, :] = targetimg_fea

        volume = volume.contiguous()
        return volume

    def build_concat_volume_reset(self, refimg_fea, targetimg_fea):
        maxdisp = self.maxdisp
        B, C, H, W = refimg_fea.shape
        volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])

        for i in range(maxdisp):
            if i > 0:
                padded_ref = refimg_fea.clone()
                padded_target = targetimg_fea.clone()
                padded_ref[:, :, :, :i] = 0  # zero first i columns
                padded_target[:, :, :, -i:] = 0  # zero last i columns
                padded_target = torch.roll(padded_target, shifts=i, dims=-1)
                concatenated = torch.cat((padded_ref, padded_target), dim=1)
                volume[:, :, i, :, :] = concatenated
            else:
                concatenated = torch.cat((refimg_fea, targetimg_fea), dim=1)
                volume[:, :, i, :, :] = concatenated

        volume = volume.contiguous()
        return volume
    
    def build_concat_volume_pad(self, refimg_fea, targetimg_fea):
        maxdisp = self.maxdisp
        B, C, H, W = refimg_fea.shape
        volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])

        for i in range(maxdisp):
            if i > 0:
                padded_ref = F.pad(refimg_fea[:, :, :, i:], (i, 0), "constant", 0)
                padded_target = F.pad(targetimg_fea[:, :, :, :-i], (i, 0), "constant", 0)
                concatenated = torch.cat((padded_ref, padded_target), dim=1)
                volume[:, :, i, :, :] = concatenated
            else:
                concatenated = torch.cat((refimg_fea, targetimg_fea), dim=1)
                volume[:, :, i, :, :] = concatenated

        volume = volume.contiguous()
        return volume
    
    def build_concat_volume_clear(self, refimg_fea, targetimg_fea):
        maxdisp = self.maxdisp
        B, C, H, W = refimg_fea.shape
        volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])

        for i in range(maxdisp):
            if i > 0:
                refimg_fea[:, :, :, i-1] = 0
                targetimg_fea[:, :, :, -1] = 0

                targetimg_fea = torch.roll(targetimg_fea, shifts=1, dims=-1)

                concatenated = torch.cat((refimg_fea, targetimg_fea), dim=1)
                volume[:, :, i, :, :] = concatenated
            else:
                concatenated = torch.cat((refimg_fea, targetimg_fea), dim=1)
                volume[:, :, i, :, :] = concatenated

        volume = volume.contiguous()
        return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out
