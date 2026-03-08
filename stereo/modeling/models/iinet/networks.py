import antialiased_cnns
from torchvision import models
import numpy as np
import timm
import torch
from torch import nn
from torchvision.ops import FeaturePyramidNetwork
import os
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from timm.models.layers import DropPath

import sys
sys.path.append('.')
from .layers import BasicBlock, multiple_basic_block
from .utils import upsample
from timm.models.mobilenetv3 import _cfg
from functools import partial

import copy

class RaftUpSampler(nn.Module):
    def __init__(
                self,
                fdim=128,
                hdim=128,
                priordim=1,
            ):
        super().__init__()
        self.fdim = fdim
        self.hdim = hdim
        self.regresshead = BasicBlock(self.fdim + priordim, self.hdim)
        self.disphead = nn.Sequential(nn.Conv2d(self.hdim, self.hdim, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hdim, 1, 3, padding=1))
        #TODO:Ablation on Context Upsampler
        self.maskhead = self.mask = nn.Sequential(
            nn.Conv2d(self.hdim, self.hdim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hdim, 4*9, 1, padding=0))

    def upsample_disp(self, disp, mask):
        """ Upsample flow field [H, W, 2] -> [2 * H/4, 2 * W, 2] using convex combination """
        N, _, H, W = disp.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(disp, [3, 3], padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, 2*H, 2*W)

    def forward(self, x, disp_l):
        net = self.regresshead(x)
        #TODO:Ablation on Residual disp
        disp_l = self.disphead(net) + disp_l
        #disp_l = self.disphead(net)
        #TODO:Ablation on Context Upsampler
        #disp_up = F.interpolate(disp_l, scale_factor=2)
        mask = .25 * self.maskhead(net)
        disp_up = self.upsample_disp(disp_l, mask)

        return disp_l, disp_up


class DepthDecoderMSR(nn.Module):
    def __init__(
                self,
                num_ch_enc,
                lrcv_scale=3,
                multiscale=1,
                scales=4,
                num_output_channels=1,
                use_skips=True
            ):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales
        self.start_scale = lrcv_scale
        self.multiscale = multiscale

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([32, 64, 128, 256])
        self.num_ch_hdim = np.array([32, 64, 128, 256])

        # decoder
        self.convs = nn.ModuleDict()
        self.refinenets = nn.ModuleDict()
        # i is encoder depth (top to bottom)
        # j is decoder depth (left to right)
        for j in range(1, 2):
            max_i = 5 - j
            for i in range(max_i, 0, -1):

                num_ch_out = self.num_ch_dec[i - 1]
                total_num_ch_in = 0

                num_ch_in = self.num_ch_enc[i - 1] if j == 1 else self.num_ch_dec[i - 1]
                self.convs[f"right_conv_{i}{j - 1}"] = BasicBlock(num_ch_in,
                                                                    num_ch_out)
                total_num_ch_in += num_ch_out

                num_ch_in = self.num_ch_dec[i] if i != 4 else self.num_ch_enc[i]
                self.convs[f"up_conv_{i + 1}{j - 1}"] = BasicBlock(num_ch_in,
                                                                    num_ch_out)
                total_num_ch_in += num_ch_out

                self.convs[f"in_conv_{i}{j}"] = multiple_basic_block(
                                                            total_num_ch_in,
                                                            num_ch_out,
                                                        )
                if 0 < i < self.scales:
                    self.refinenets[f"output_{i}"] = RaftUpSampler(fdim=num_ch_out,
                                                                   hdim=self.num_ch_hdim[i - 1],
                                                                   priordim=1)

    def forward(self, input_features, priority):
        prev_outputs = input_features
        depth_outputs = {}
        prev_dnc = {'disp': priority['cdisp'],
                    'conf': priority['cconf']}
        stage = len(priority['cdisp']) - 1
        up_disp = None
        for j in range(1, 2):
            max_i = 5 - j # 4
            upfeats = []
            for i in range(max_i, 0, -1):

                inputs = [self.convs[f"right_conv_{i}{j - 1}"](prev_outputs[i - 1])]

                if i == max_i: # 4: up_conv_50
                    inputs += [upsample(self.convs[f"up_conv_{i + 1}{j - 1}"](prev_outputs[i]))]
                else: # 3, 2, 1: up_conv_41, up_conv_31, up_conv_21
                    inputs += [upsample(self.convs[f"up_conv_{i + 1}{j - 1}"](upfeats[-1]))]

                upfeat = self.convs[f"in_conv_{i}{j}"](torch.cat(inputs, dim=1))
                upfeats += [upfeat]

                if 0 < i < self.scales:
                    if i == self.start_scale:
                        displ, up_disp = self.refinenets[f"output_{i}"](
                           torch.cat([upfeat, prev_dnc['disp'][stage]], dim=1),prev_dnc['disp'][stage])
                        stage -= 1
                    else:
                        displ, up_disp = self.refinenets[f"output_{i}"](
                           torch.cat([upfeat, up_disp], dim=1), up_disp)
                        stage -= 1

                    depth_outputs[f"disp_pred_s{i}"] = displ

            prev_outputs = upfeats[::-1]
            depth_outputs["disp_pred"] = up_disp
            # depth_outputs["disp_pred_s0"] = up_disp

        return depth_outputs


class CVEncoder(nn.Module):
    def __init__(self, num_ch_cv, num_ch_encs, num_ch_outs, lrcv_level, multi_scale=1):
        super().__init__()

        self.convs = nn.ModuleDict()
        self.num_ch_enc = []
        self.multi_scale = multi_scale
        start_level = lrcv_level - multi_scale
        num_ch_subencs = num_ch_encs[start_level:] #lrcv_level=0 represent stat with (1/2) resolution
        num_ch_subouts = num_ch_outs[start_level:]

        self.num_blocks = len(num_ch_subouts)
        num_ch_cvs = [4, 6, num_ch_cv]
        num_ch_cvs = num_ch_cvs[start_level:]

        for i in range(self.num_blocks): # 0, 1, 2, 3, 4
            num_ch_in = num_ch_subouts[i - 1]
            num_ch_out = num_ch_subouts[i]
            if i == 0:
                num_ch_fuse = num_ch_cvs[i] + num_ch_subencs[i]
            elif i < multi_scale + 1: # 1, 2
                self.convs[f"ds_conv_{i}"] = BasicBlock(num_ch_in, num_ch_out, stride=2)
                num_ch_fuse = num_ch_cvs[i] + num_ch_out + num_ch_subencs[i]
            else: # 3, 4
                self.convs[f"ds_conv_{i}"] = BasicBlock(num_ch_in, num_ch_out, stride=2)
                num_ch_fuse = num_ch_out + num_ch_subencs[i]
            self.convs[f"conv_{i}"] = nn.Sequential( # 0, 1, 2, 3, 4
                BasicBlock(num_ch_fuse, num_ch_out, stride=1),
                BasicBlock(num_ch_out, num_ch_out, stride=1),
            )
            self.num_ch_enc.append(num_ch_out)

    def forward(self, cost_list, img_feats):
        num_stage = len(cost_list) # for MULTISCALE = 2, MATCHING_SCALE = 2 => num_stage = 3

        outputs = [None] * self.num_blocks
        cost = cost_list[0]
        x = torch.cat([cost, img_feats[0]], dim=1) # cost [B, D_bins, H, W], feats [B, C_i, H, W]
        x = self.convs[f"conv_{0}"](x)
        outputs[0] = x
        for i in range(1, self.num_blocks):
            x = self.convs[f"ds_conv_{i}"](x)
            if i < num_stage: # 1, 2
                cost = cost_list[i]
                x = torch.cat([cost, x, img_feats[i]], dim=1)
                x = self.convs[f"conv_{i}"](x)
            else: # 3, 4
                x = torch.cat([x, img_feats[i]], dim=1)
                x = self.convs[f"conv_{i}"](x)
            outputs[i] = x

        return outputs


class MLP(nn.Module):
    def __init__(self, channel_list, disable_final_activation=True):
        super(MLP, self).__init__()

        layer_list = []
        for layer_index in list(range(len(channel_list)))[:-1]:
            layer_list.append(
                            nn.Linear(channel_list[layer_index], 
                                channel_list[layer_index+1])
                            )
            layer_list.append(nn.LeakyReLU(inplace=True))

        if disable_final_activation:
            layer_list = layer_list[:-1]

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)


class ResnetMatchingEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(
                self,
                num_layers,
                num_ch_out,
                matching_scale=1,
                multiscale=1,
                pretrainedfp=None,
                antialiased=True,
            ):
        super().__init__()

        if matching_scale == 1:
            self.num_ch_enc = np.array([64, 64])
        elif matching_scale == 2:
            self.num_ch_enc = np.array([64, 128])

        model_source = antialiased_cnns if antialiased else models

        resnet = model_source.resnet18


        if num_layers !=18:
            raise ValueError("{} is not a valid number of resnet layers"
                                                            .format(num_layers))

        encoder = resnet(True)
        #pretrained_model = torch.load(pretrainedfp)['state_dict']
        #encoder.load_state_dict(pretrained_model)

        resnet_backbone_four = [
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
            encoder.layer1,
        ]
        self.net = nn.Sequential(*resnet_backbone_four)
        if matching_scale == 2:
            self.net1 = nn.Sequential(*[encoder.layer2])

        self.num_ch_out = num_ch_out

        self.head = nn.Sequential(
            nn.Conv2d(self.num_ch_enc[0], 64, (1, 1)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                    64,
                    self.num_ch_out,
                    (3, 3),
                    padding=1,
                    padding_mode="replicate"
                ),
            nn.InstanceNorm2d(self.num_ch_out)
        )
        if matching_scale == 2:
            self.head1 = nn.Sequential(
            nn.Conv2d(self.num_ch_enc[1], 128, (1, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                    128,
                    self.num_ch_out,
                    (3, 3),
                    padding=1,
                    padding_mode="replicate"
                ),
            nn.InstanceNorm2d(self.num_ch_out)
        )
        del encoder

        self.multiscale = multiscale

    def forward(self, x):
        output = []
        x = self.net(x)
        if self.multiscale == 1:
            output.append(self.head(x))
        if hasattr(self, 'net1'):
            x = self.net1(x)
            output.append(self.head1(x))
        else:
            output.append(self.head(x))
        return output


class UnetMatchingEncoder(nn.Module):
    """Pytorch module for a resnet encoder
        """

    def __init__(
            self,
            num_ch_out,
            matching_scale=1,
            multiscale=1,
            pretrainedfp=None,
            cfgs = None
    ):
        super().__init__()

        self.backbone_cfgs = cfgs

        self.num_ch_enc = np.array([16, 24, 40, 112, 160])
        self.num_ch_up = np.array([16, 24, 40, 112])
        self.multiscale = multiscale
        # self.num_ch_out = np.array([16, 16, 16, 16])
        self.num_ch_out = np.ones((4)) * num_ch_out # defined in yaml file
        self.lrcvscale = matching_scale + 1

        if pretrainedfp:
            config = _cfg(url='', file=pretrainedfp)

            model = timm.create_model(
                "mobilenetv3_large_100",
                pretrained=True,
                features_only=True,
                pretrained_cfg=config
            )
        else:
            model = timm.create_model(
                "mobilenetv3_large_100",
                pretrained=True,
                features_only=True,
            )
        self.stage0 = nn.Sequential(
            model.conv_stem,
            model.bn1,
            model.act1
        )

        layers = [1, 2, 3, 5, 6]
        self.stage1 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.stage2 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.stage3 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.stage4 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.stage5 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])


        self.convs = nn.ModuleDict()

        for i in range(1, 5):
            num_ch_up = self.num_ch_enc[i]
            num_ch_left = self.num_ch_enc[i - 1]
            num_ch_right = self.num_ch_up[i - 1]
            num_ch_out_l = int(self.num_ch_out[i - 1])
            self.convs[f'in_conv{i}'] = nn.Sequential(
                nn.Conv2d(num_ch_left + num_ch_right,  num_ch_right, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_ch_right),
                nn.LeakyReLU(0.2, True))
            if self.lrcvscale - multiscale <= i < self.lrcvscale + 1:
                self.convs[f'out_conv{i}'] = nn.Sequential(
                    nn.Conv2d(num_ch_right, num_ch_out_l, kernel_size=3, padding=1, padding_mode="replicate"),
                    nn.InstanceNorm2d(num_ch_out_l))
            self.convs[f'up_conv{i+1}'] = nn.Sequential(
                nn.ConvTranspose2d(num_ch_up, num_ch_right, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_ch_right),
                nn.LeakyReLU(0.2, True))

        del model

    def forward(self, x):
        output = [None] * (self.multiscale + 1) # [None, None, None]
        enc_output = [None] * 5
        feat_output = [None] * 5
        x = self.stage1(self.stage0(x))
        enc_output[0] = x
        feat_output[0] = x
        x = self.stage2(x)
        enc_output[1] = x
        feat_output[1] = x
        x = self.stage3(x)
        enc_output[2] = x
        feat_output[2] = x
        x = self.stage4(x)
        enc_output[3] = x
        feat_output[3] = x
        x = self.stage5(x)
        enc_output[4] = x
        feat_output[4] = x
        # decoder part
        for i in range(4, 0, -1): # i = 4, 3, 2, 1
            x_right = enc_output[i - 1]
            x_diag = self.convs[f'up_conv{i + 1}'](enc_output[i])
            x_up = self.convs[f'in_conv{i}'](torch.cat([x_right, x_diag], dim=1))
            feat_output[i - 1] = x_up
            if self.lrcvscale - self.multiscale <= i <= self.lrcvscale: # if 1, 2, or 3
                output[i - self.lrcvscale + self.multiscale] = self.convs[f'out_conv{i}'](x_up)
        return output, feat_output


class ConcatenatedUnetMatchingEncoder(UnetMatchingEncoder):
    def __init__(
            self,
            num_ch_out,
            matching_scale=1,
            multiscale=1,
            pretrainedfp=None,
            cfgs = None
    ):
        super().__init__(
            num_ch_out,
            matching_scale,
            multiscale,
            pretrainedfp,
            cfgs
        )
        self.backbone_cfgs = cfgs

        self.impl = super().forward

        # default implementation
        self.forward = self.forward_left_right_one_by_one

        if (cfgs is not None):
            should_concat = cfgs.get('CONCAT_LEFT_RIGHT', False) # if not supplied, then assume False
            if should_concat:
                concat_type = cfgs.get('CONCAT_LEFT_RIGHT_ALONG', None)
                if concat_type is None:
                    raise ValueError(f"Could not find MODEL.BACKBONE_CFGS.CONCAT_LEFT_RIGHT_ALONG parameter in yaml config file.")
                print(f"BACKBONE CONCAT LEFT-RIGHT is ENABLED. Type: {concat_type}")
                if concat_type == 'batch':
                    self.forward = self.forward_left_right_images_along_batch
                elif concat_type == 'horizontal':
                    self.forward = self.forward_left_right_images_horizontal
                elif concat_type == 'vertical':
                    self.forward = self.forward_left_right_images_vertical
                elif concat_type == 'multicut':
                    self.forward = self.forward_left_right_images_multicut
                    self.h_division = cfgs.get('H_DIVISION', None)
                    self.w_division = cfgs.get('W_DIVISION', None)
                    if self.h_division is None or self.w_division is None:
                        raise ValueError(f"MODEL.BACKBONE_CFGS.H_DIV and W_DIV must be specified for multicut concatenation.")
                    print(f"Multicut concat: H_DIVISION={self.h_division}, W_DIVISION={self.w_division}")
                else:
                    raise NotImplementedError(f"Concat type '{concat_type}' is not implemented. Available: batch, horizontal, vertical, multicut")


    def forward_left_right_one_by_one(self, left, right):
        features_left = self.impl(left)
        features_right = self.impl(right)
        return features_left, features_right

    def forward_left_right_images_along_batch(self, left, right):
        # concatenate along batch dimension
        combined = torch.cat([left, right], dim=0)

        # pass through backbone
        combined_output, combined_feat_output = self.impl(combined)
        # returns output, feat_output

        # # split features back to left and right
        
        output_left = [feat[:left.size(0)] for feat in combined_output]
        output_right = [feat[left.size(0):] for feat in combined_output]
        
        feat_output_left = [feat[:left.size(0)] for feat in combined_feat_output]
        feat_output_right = [feat[left.size(0):] for feat in combined_feat_output]

        return (output_left, feat_output_left), (output_right, feat_output_right)

    def forward_left_right_images_horizontal(self, left, right):
        # concatenate left and right images along width dimension
        concat_images = torch.cat((left, right), dim=-1)  # [B, C, H, 2*W]

        output, feat_output = self.impl(concat_images)

        output_left = []
        output_right = []

        for feat in output:
            w = feat.size(-1)
            w_divided = w // 2
            output_left.append(feat[..., :w_divided])
            output_right.append(feat[..., w_divided:])

        feat_output_left = []
        feat_output_right = []

        for feat in feat_output:
            w = feat.size(-1)
            w_divided = w // 2
            feat_output_left.append(feat[..., :w_divided])
            feat_output_right.append(feat[..., w_divided:])

        return (output_left, feat_output_left), (output_right, feat_output_right)

    def forward_left_right_images_vertical(self, left, right):
        # concatenate along vertical dimension
        combined = torch.cat([left, right], dim=-2)

        # pass through backbone
        output, feat_output = self.impl(combined)

        output_left = []
        output_right = []

        # split features back to left and right
        for feat in output:
            h = feat.size(-2)
            h_divided = h // 2
            output_left.append(feat[..., :h_divided, :])
            output_right.append(feat[..., h_divided:, :])

        feat_output_left = []
        feat_output_right = []

        for feat in feat_output:
            w = feat.size(-2)
            w_divided = w // 2
            feat_output_left.append(feat[..., :w_divided, :])
            feat_output_right.append(feat[..., w_divided:, :])

        return (output_left, feat_output_left), (output_right, feat_output_right)

    def forward_left_right_images_multicut(self, image_left, image_right):
        
        # [24, 3, 544, 960], tensor contigous = yes type: <class 'torch.Tensor'>

        # cut each image into 4 parts vertically and 2 parts horizontally -> 8 parts
        image_left_parts = []
        image_right_parts = []

        h_division = self.h_division
        w_division = self.w_division
        # after division, the new height and width need to be divisible by 32 (depending on the backbone)

        nr_of_parts = w_division * h_division  # 8
        h, w, b = image_left.size(-2), image_left.size(-1), image_left.size(0)
        h_divided = h // h_division
        w_divided = w // w_division
        assert h_divided % 32 == 0, f"Divided down height {h_divided} not divisible by 32 (originally {h})"
        assert w_divided % 32 == 0, f"Divided down width {w_divided} not divisible by 32 (originally {w})"
        
        # print(f"Dividing images into {nr_of_parts} parts: {h_division} along height, {w_division} along width. Each part size: {h_divided} x {w_divided} from original {h} x {w}. Batch size is {b} so we expect {nr_of_parts * b} parts total along batch dimension.")

        for i in range(h_division):
            for j in range(w_division):
                image_left_parts.append(image_left[..., i*h_divided:(i+1)*h_divided, j*w_divided:(j+1)*w_divided])
                image_right_parts.append(image_right[..., i*h_divided:(i+1)*h_divided, j*w_divided:(j+1)*w_divided])
        
        left_stack = torch.cat(image_left_parts, dim=0)
        right_stack = torch.cat(image_right_parts, dim=0)

        # concatenate all parts along batch dimension
        combined = torch.cat((left_stack, right_stack), dim=0) 

        # print(f"Combined parts shape: {combined.shape}, should be [{nr_of_parts * b * 2}, c, h/div_h, w/div_v]" )

        output, feat_output = self.impl(combined)

        output_left = []
        output_right = []

        # handle 'output' first
        for feat in output:

            # should be [b*2*8, c, h/4, w/2]
            combined_features_left = feat[:nr_of_parts*b, ...]
            combined_features_right = feat[nr_of_parts*b:, ...]

            left_feat_image = []
            right_feat_image = []
            
            for bi in range(b):

                left_v_sequence = []
                right_v_sequence = []
                for i in range(h_division):
                
                    left_h_sequence = []
                    right_h_sequence = []
                    
                    for j in range(w_division):
                    
                        left_h_sequence.append(combined_features_left[i*b*w_division + j*b + bi, ...]) # append to horizontal sequence
                        right_h_sequence.append(combined_features_right[i*b*w_division + j*b + bi, ...])

                    left_v_sequence.append(torch.cat(left_h_sequence, dim=-1)) # first concat along width (get full row), then append to vertical sequence
                    right_v_sequence.append(torch.cat(right_h_sequence, dim=-1))

                left_feat_image.append(torch.cat(left_v_sequence, dim=-2)) # first concat along height (get full image), then append along batch dimension
                right_feat_image.append(torch.cat(right_v_sequence, dim=-2))


            output_left.append(torch.stack(left_feat_image, dim=0)) # first stack along batch, then append to features list
            output_right.append(torch.stack(right_feat_image, dim=0))

        feat_output_left = []
        feat_output_right = []

        # handle 'feat_output' second
        for feat in feat_output:

            # should be [b*2*8, c, h/4, w/2]
            combined_features_left = feat[:nr_of_parts*b, ...]
            combined_features_right = feat[nr_of_parts*b:, ...]

            left_feat_image = []
            right_feat_image = []
            
            for bi in range(b):

                left_v_sequence = []
                right_v_sequence = []
                
                for i in range(h_division): # along height

                    left_h_sequence = []
                    right_h_sequence = []
                
                    for j in range(w_division): # along width

                        left_h_sequence.append(combined_features_left[i*b*w_division + j*b + bi, ...]) # append to horizontal sequence
                        right_h_sequence.append(combined_features_right[i*b*w_division + j*b + bi, ...])

                    left_v_sequence.append(torch.cat(left_h_sequence, dim=-1)) # first concat along width (get full row), then append to vertical sequence
                    right_v_sequence.append(torch.cat(right_h_sequence, dim=-1))

                left_feat_image.append(torch.cat(left_v_sequence, dim=-2)) # first concat along height (get full image), then append along batch dimension
                right_feat_image.append(torch.cat(right_v_sequence, dim=-2))

            feat_output_left.append(torch.stack(left_feat_image, dim=0)) # first stack along batch, then append to features list
            feat_output_right.append(torch.stack(right_feat_image, dim=0))


        return (output_left, feat_output_left), (output_right, feat_output_right)
    
