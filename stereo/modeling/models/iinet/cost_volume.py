import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

import sys
sys.path.append('.')

from .networks import MLP

import math


class MsCostVolumeManager(nn.Module):
    def __init__(self,
                 num_depth_bins=64,
                 multiscale=1,
                 disp_scale=2,
                 matching_dim_size=16,
                 dot_dim=1):
        super().__init__()
        self.num_depth_bins = [4, 6, num_depth_bins] # 4, 6, 24 in our case
        self.disp_scale = disp_scale
        self.stride = [1, 2, 4, 8]
        self.num_cv = multiscale + 1 # 2 + 1 = 3 in our case
        self.num_depth_bins = self.num_depth_bins[-self.num_cv:]
        self.stride = self.stride[:self.num_cv][::-1]
        self.in_channels = 2 * matching_dim_size + 1 # 2 * 16 + 1 = 33
        self.feat_scale = np.sqrt(matching_dim_size) # 4.0
        channel_list = [self.in_channels, 64, 32, 1]
        self.mlp = MLP(channel_list=channel_list)
        self.beta = - np.log(0.5)
        self.pt = 0.90
        self.alpha = 1 / (-self.pt * np.log(self.pt) - (1 - self.pt) * np.log(1 - self.pt) - self.beta)

    def forward(self, left_feats, right_feats):
        """ Runs the cost volume and gets the lowest cost result """

        num_stage = len(left_feats) # for MULTISCALE = 2, MATCHING_SCALE = 2 => num_stage = 3

        # loop through depth planes
        cost_volume = [None] * num_stage
        confidence = [None] * num_stage

        coarse_disp = [None] * num_stage
        hypos = [None] * (num_stage - 1)
        
        for k in range(num_stage - 1, -1, -1): # from coarse to fine (low-res to high-res)
            batch_size, num_feat_channels, matching_height, matching_width = left_feats[k].shape
            volume = left_feats[k].new_zeros([batch_size, self.num_depth_bins[k], matching_height, matching_width])

            if k == num_stage - 1: # the coarsest stage

                dot_feat = torch.sum(left_feats[k] * right_feats[k], dim=1, keepdim=True) / self.feat_scale
                mlp_feat = torch.cat((left_feats[k][:, :, :, :], right_feats[k][:, :, :, :], dot_feat), dim=1)
                volume[:, 0, :, :] = self.mlp(mlp_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).squeeze(1)
                for depth_id in range(1, self.num_depth_bins[k]):
                    mlp_feat = left_feats[0].new_zeros([batch_size, self.in_channels, matching_height, matching_width])
                    mlp_feat[:, :num_feat_channels] = left_feats[k] # take existing left features
                    mlp_feat[:, num_feat_channels:2*num_feat_channels, :, depth_id:] = right_feats[k][:, :, :, :-depth_id] # take only the right feats that are applicable for current depth and add them to mlp_feat
                    mlp_feat[:, -1:, :, depth_id:] = torch.sum(left_feats[k][:, :, :, depth_id:] * right_feats[k][:, :, :, :-depth_id], dim=1, keepdim=True) / self.feat_scale # dot product(element wise mul+sum), normalized
                    volume[:, depth_id, :, :] = self.mlp(mlp_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).squeeze(1) # [B, D, H, W] => [B, 1, H, W]

                cost_volume[k] = volume # [B, D, H, W] where D = num_depth_bins[2] = 24
                p = torch.sigmoid(cost_volume[k] * 2).clamp(min=1e-7, max=1 - 1e-7) # not softmax
                q = 1 - p

                entropy = - p * torch.log(p) - q * torch.log(q) # low entropy means high confidence in accuracy of disparity candidate (being either the BEST or WORST candidate)
                uncertainty = torch.mean(entropy, dim=1, keepdim=True) # if all are low (all are confident), then uncertainty is low
                # if one is low (high confidence) but other are high (low confidence), then uncertainty is high
                confidence[k] = torch.clamp(
                    self.alpha * (uncertainty - self.beta) * (torch.max(p, dim=1)[0].unsqueeze(1)), max=1.0) # multiplied with the max probability of the best candidate
                prob_volume = torch.softmax(volume, dim=1)
                _, ind = prob_volume.sort(dim=1, descending=True)
                coarse_disp[k] = ind[:, :1].float() * (confidence[k] > 0.1) / self.disp_scale
                if len(hypos) > 0:
                    hypos[k - 1] = F.interpolate(ind[:, :self.num_depth_bins[k - 1] // 2].float(), scale_factor=2) 
                    # by default mode = 'nearest', so it will not interpolate, but just repeat the values
                    # it increase in the spatial dimension, but not in the depth dimension
                    # D // 2 = num_depth_bins[1] // 2 = 6 // 2 = 3, 
                    # D // 2 = num_depth_bins[0] // 2 = 4 // 2 = 2  

            else:

                mlp_feat = left_feats[0].new_zeros(
                   [batch_size, self.in_channels, self.num_depth_bins[k] * matching_height, matching_width])

                # D = num_depth_bins[1] = 6, num_depth_bins[0] = 4

                prev_ind = hypos[k]  # b, d//2,h ,w

                hpos = torch.arange(0, matching_height, device=volume.device).view(1, 1, matching_height, 1).expand(
                    batch_size, self.num_depth_bins[k], matching_height, matching_width)
                wpos = torch.arange(0, matching_width, device=volume.device).view(1, 1, 1, matching_width)  # [1, 1, 1, W]
                # If previous disparity index was d, now consider 2d and 2d+1
                dpos = torch.stack([prev_ind * 2, prev_ind * 2 + 1], dim=2).view(batch_size, -1, matching_height,
                                                                                 matching_width) # [B, D, H, W]
                hypos[k] = dpos

                tgt_wpos = wpos - dpos  # b,d,h,w
                # normalize to [-1, 1]
                coords_x = tgt_wpos / ((matching_width - 1) / 2) - 1
                coords_y = hpos / ((matching_height - 1) / 2) - 1
                grid = torch.stack([coords_x, coords_y], dim=4) # [B, D, H, W, 2]
                # grid_sample, takes for num_depth_bins[k] differenet disparity values right_feats[k] offset by dpos
                # so not every disparity value is sampled in for loop, but only num_depth_bins[k] of them:
                # it takes offset-ed right_feats and puts them in the expanded height axis
                tgt_feats = F.grid_sample(right_feats[k],
                                          grid.view(batch_size, self.num_depth_bins[k] * matching_height,
                                                    matching_width, 2),
                                          mode='nearest',
                                          padding_mode='zeros',
                                          align_corners=False) # [B, C, D*H, W]
                left_feat = left_feats[k].repeat(1, 1, self.num_depth_bins[k], 1) # [B, C, D*H, W]
                mlp_feat[:, :num_feat_channels] = left_feat
                mlp_feat[:, num_feat_channels:2*num_feat_channels] = tgt_feats
                mlp_feat[:, -1:] = torch.sum(left_feat * tgt_feats, dim=1, keepdim=True) / self.feat_scale
                # before mlp_feat: [B, C, D*H, W] => [B, D*H, W, C]
                # after mlp_feat: [B, D*H, W, 1] => [B, 1, D*H, W]
                # reshape => [B, depth_bins, H, W]
                volume = self.mlp(mlp_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).reshape(batch_size,
                                                                                            self.num_depth_bins[k],
                                                                                            matching_height,
                                                                                            matching_width)

                cost_volume[k] = volume

                p = torch.sigmoid(cost_volume[k] * 2).clamp(min=1e-7, max=1 - 1e-7)
                q = 1 - p
                entropy = - p * torch.log(p) - q * torch.log(q)
                uncertainty = torch.mean(entropy, dim=1, keepdim=True)
                confidence[k] = torch.clamp(
                    self.alpha * (uncertainty - self.beta) * (torch.max(p, dim=1)[0].unsqueeze(1)), max=1.0)
                max_ind = torch.max(volume, dim=1)[1].unsqueeze(1) # get the index of the depth bin with the max value
                # TODO:Ablation on filter
                # use gather to get the actual disparity value for the max index
                coarse_disp[k] = torch.gather(dpos, 1, max_ind) * (confidence[k] > 0.1) / (self.disp_scale * 2 ** (num_stage - 1 - k))
                if k > 0:
                    _, ind = volume.sort(dim=1, descending=True)
                    hypos[k - 1] = torch.gather(dpos, 1, ind[:, :self.num_depth_bins[k - 1] // 2])
                    hypos[k - 1] = F.interpolate(hypos[k - 1].float(), scale_factor=2)

        # cost_volume: 
        # k=2 [B, 24, H, W], <- not max disp 192 because it is done on subsampled image (1/8)
        # k=1 [B, 6, H/2, W/2], <- subsampled image (1/4)
        # k=0 [B, 4, H/4, W/4], <- subsampled image (1/2)
        return cost_volume, hypos, {'cdisp': coarse_disp, 'cconf': confidence}



