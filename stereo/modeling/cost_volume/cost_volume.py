# @Time    : 2023/10/8 05:02
# @Author  : zhangchenming
import numpy as np
import torch
import torch.nn as nn
from stereo.modeling.common.basic_block_3d import BasicConv3d
from stereo.modeling.common.basic_block_2d import BasicConv2d

# class Simple4DVolume(nn.Module):
# function of this class returns calculated cost volume
# which means it should be called from a forward() function
# of a model 
# 4D = [batch, features, dispariry, width, height]
class SimpleCorrelationVolumes:
    def __init__(self, type, group=1):
        self.group = group
        if type == 'SIMPLE_3D':
            self.volume = SimpleCorrelationVolume_3D(group)
            # self.volume = SimpleCorrelationVolume_3D()
        elif type == 'SIMPLE_4D':
            self.volume = SimpleCorrelationVolume_4D(group)
        elif type == 'ALL_PAIRS_4D':
            self.volume = Simple_AllPairsVolume_4D(group)
        elif type == 'ALL_PAIRS_3D':
            self.volume = Simple_AllPairsVolume_3D(group)
        else:
            raise ValueError(f"Unknown type: {type}. Supported types are 'SIMPLE_3D', 'SIMPLE_4D', 'ALL_PAIRS_4D', 'ALL_PAIRS_3D'.")
        
    def calculate(self, left_feat, right_feat, max_disp = None):
        return self.volume.calculate(left_feat, right_feat, max_disp)
    

    

class Simple_AllPairsVolume_4D:
    def __init__(self, group=1):
        # super(Simple4DVolume, self).__init__()
        pass

    def calculate(self, left_feat, right_feat, max_disp = None):
        return self.calculate_allpairs(left_feat, right_feat)

    def calculate_allpairs(self, left_feat, right_feat):
        b, f, h, y = left_feat.shape
        b, f, h, z = right_feat.shape
        disparity_dim = z
        # all pairs correlation
        overlap = np.einsum('bfhy,bfhz->bfhyz', left_feat, right_feat) 
        # shape [b, f, h, w_left, w_right] = [b, f, h, w_left, disparity]
        # 
        print(f"overlap shape before permute (4D): {overlap.shape}")
        overlap = overlap.permute(b, f, disparity_dim, h, y) 

        print(f"overlap shape after permute (4D): {overlap.shape}")

        return overlap[:, :, :self.maxdisp, :, :].contiguous()  # return only maxdisp disparity planes
    
class Simple_AllPairsVolume_3D:
    def __init__(self, group=1):
        # super(Simple4DVolume, self).__init__()
        pass

    def calculate(self, left_feat, right_feat, max_disp):
        return self.calculate_allpairs(left_feat, right_feat, max_disp)

    def calculate_allpairs(self, left_feat, right_feat, max_disp):
        b, f, h, y = left_feat.shape
        b, f, h, z = right_feat.shape
        disparity_dim = z
        # all pairs correlation
        overlap = np.einsum('bfhy,bfhz->bhyz', left_feat, right_feat)

        print(f"overlap shape before permute (3D): {overlap.shape}")
        overlap = overlap.permute(b, disparity_dim, h, y)
        print(f"overlap shape after permute (3D): {overlap.shape}")

        return overlap[:, :max_disp, :, :].contiguous()  # return only maxdisp disparity planes
    
class SimpleCorrelationVolume_3D: # default
    def __init__(self, group=1):
        self.group = group

    def calculate(self, left_feat, right_feat, max_disp):
        # return self.correlation_volume(left_feat, right_feat, max_disp)
        return self.correlation_volume_roll(left_feat, right_feat, max_disp)

    def correlation_volume(self, left_feature, right_feature, max_disp):
        b, c, h, w = left_feature.size()
        cost_volume = left_feature.new_zeros(b, max_disp, h, w)
        for i in range(max_disp):
            if i > 0:
                cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] * right_feature[:, :, :, :-i]).mean(dim=1)
            else:
                cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
        cost_volume = cost_volume.contiguous()
        return cost_volume # [b, max_disp, h, w]
    
    # removes indexing
    def correlation_volume_vectorized(self, left_feature, right_feature, max_disp):
        """
        Vectorized cost volume:
        left_feature, right_feature: [B, C, H, W]
        Returns: [B, max_disp, H, W]
        """
        B, C, H, W = left_feature.shape
        # Build all shifted right features with left padding (zero)
        # right_shifted[d]: right shifted left by d (equivalent to original[:, :, :, :-d] aligned with left[:, :, :, d:])
        shifted_list = []
        for d in range(max_disp):
            if d == 0:
                shifted = right_feature
            else:
                # pad on right side to keep W, then remove last d columns to emulate shift
                # Equivalent to left padding of d zeros
                shifted = torch.pad(right_feature[..., :-d], (d, 0), mode='constant', value=0)
            shifted_list.append(shifted)
        # [B, C, H, W] -> stack -> [B, max_disp, C, H, W]
        right_stack = torch.stack(shifted_list, dim=1)
        # Expand left for broadcast: [B, 1, C, H, W] -> [B, max_disp, C, H, W]
        left_exp = left_feature.unsqueeze(1).expand_as(right_stack)
        prod = left_exp * right_stack  # [B, max_disp, C, H, W]
        # Mask invalid columns (those padded)
        if W >= max_disp:
            cols = torch.arange(W, device=left_feature.device).view(1, 1, 1, 1, W)
            disp = torch.arange(max_disp, device=left_feature.device).view(1, max_disp, 1, 1, 1)
            valid = (cols >= disp)  # broadcast
            prod = prod * valid  # zero out invalid region
        cost = prod.mean(dim=2)  # channel mean -> [B, max_disp, H, W]
        return cost.contiguous()

    # alternative implementation using torch.roll (lower memory)
    def correlation_volume_roll(self, left_feature, right_feature, max_disp):
        B, C, H, W = left_feature.shape
        disp_slices = []
        cols = torch.arange(W, device=left_feature.device)
        for d in range(max_disp):
            shifted = torch.roll(right_feature, shifts=d, dims=-1)
            prod = left_feature * shifted
            valid_mask = (cols >= d).view(1, 1, W)
            disp_slices.append((prod.mean(1) * valid_mask))  # [B,H,W]
        return torch.stack(disp_slices, dim=1).contiguous() # actually returns [1, max_disp, batch, height, width]

class SimpleCorrelationVolume_4D:
    def __init__(self,  group=1):
        self.group = group

    def calculate(self, left_feat, right_feat, max_disp):
        return correlation_volume(left_feat, right_feat, max_disp)

    def correlation_volume(left_feature, right_feature, max_disp):
        b, c, h, w = left_feature.size()
        cost_volume = left_feature.new_zeros(b, c, max_disp, h, w)
        for i in range(max_disp):
            if i > 0:
                cost_volume[:, :, i, :, i:] = (left_feature[:, :, :, i:] * right_feature[:, :, :, :-i])
            else:
                cost_volume[:, :, i, :, :] = (left_feature * right_feature)
        cost_volume = cost_volume.contiguous()
        return cost_volume # [b, c, max_disp, h, w]
    
class Infer3DbyMLP(nn.Module):
    """
    MLP-based processing on 3D cost volume.
    Applies MLPs across the channel dimension at each spatial location (depth, height, width).
    Reduces channel dimension from input channels to 1.
    Input: [batch, channels, depth, height, width] -> Output: [batch, 1, depth, height, width]
    
    Args:
        input_channels (int): Number of input channels
        hidden_dim (list): List of hidden dimensions for each stage
        activation (str): Activation function ('relu', 'leaky_relu', 'gelu')
        dropout (float): Dropout probability (0.0 to disable)
        normalization (str): Normalization type ('layer', 'batch', 'instance', or None)
    """
    
    def __init__(self, input_channels, hidden_dim, activation='relu', dropout=0.0, 
                 normalization='layer'):
        super(Infer3DbyMLP, self).__init__()
        
        # Validate inputs
        if not isinstance(hidden_dim, (list, tuple)):
            raise ValueError("hidden_dim must be a list or tuple specifying dimensions for each stage")
        
        if len(hidden_dim) == 0:
            raise ValueError("hidden_dim cannot be empty")
        
        self.input_channels = input_channels
        self.hidden_dim = list(hidden_dim)
        self.mlp_stages = len(hidden_dim)
        
        # Build MLP layers
        layers = []
        
        # Build layers based on hidden_dim array
        prev_dim = input_channels
        
        for i, curr_dim in enumerate(self.hidden_dim):
            # Add linear layer
            layers.append(nn.Linear(prev_dim, curr_dim))
            
            # Add normalization, activation, and dropout (except possibly for the last layer)
            if i < self.mlp_stages - 1:  # Not the last layer
                # Add normalization
                if normalization == 'layer':
                    layers.append(nn.LayerNorm(curr_dim))
                elif normalization == 'batch':
                    layers.append(nn.BatchNorm1d(curr_dim))
                elif normalization == 'instance':
                    layers.append(nn.InstanceNorm1d(curr_dim))
                # elif normalization is None: no normalization

                layers.append(self._get_activation(activation))
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
            else:  # Last layer
                # For the last layer, optionally add normalization but typically no activation
                if normalization == 'layer':
                    layers.append(nn.LayerNorm(curr_dim))
                elif normalization == 'batch':
                    layers.append(nn.BatchNorm1d(curr_dim))
                elif normalization == 'instance':
                    layers.append(nn.InstanceNorm1d(curr_dim))
            
            prev_dim = curr_dim
        
        # Ensure the last layer outputs 1 dimension
        if self.hidden_dim[-1] != 1:
            layers.append(nn.Linear(self.hidden_dim[-1], 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def _get_activation(self, activation):
        """Get activation function by name"""
        if activation.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif activation.lower() == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch, channels, depth, height, width]
            
        Returns:
            Output tensor of shape [batch, 1, depth, height, width]
        """
        batch, channels, depth, height, width = x.shape
        
        # Reshape to process each spatial location (d,h,w) independently
        # [batch, channels, depth, height, width] -> [batch*depth*height*width, channels]
        x_reshaped = x.permute(0, 2, 3, 4, 1).contiguous()  # [batch, depth, height, width, channels]
        x_reshaped = x_reshaped.view(-1, channels)  # [batch*depth*height*width, channels]
        
        # Apply MLP to each vector along the channel dimension
        output = self.mlp(x_reshaped)  # [batch*depth*height*width, 1]
        
        # Reshape back to original spatial dimensions
        output = output.view(batch, depth, height, width, 1)  # [batch, depth, height, width, 1]
        output = output.permute(0, 4, 1, 2, 3).contiguous()  # [batch, 1, depth, height, width]
        
        return output




class CoExCostVolume(nn.Module):
    def __init__(self, maxdisp, group=1):
        super(CoExCostVolume, self).__init__()
        self.maxdisp = maxdisp + 1
        self.group = group
        self.unfold = nn.Unfold((1, maxdisp + 1), 1, 0, 1)
        self.left_pad = nn.ZeroPad2d((maxdisp, 0, 0, 0))

    def forward(self, x, y):
        b, c, h, w = x.shape

        y = self.left_pad(y)
        unfolded_y = self.unfold(y)
        unfolded_y = unfolded_y.reshape(b, self.group, c // self.group, self.maxdisp, h, w)

        x = x.reshape(b, self.group, c // self.group, 1, h, w)

        cost = (x * unfolded_y).sum(2)
        cost = torch.flip(cost, dims=[2])

        return cost


def correlation_volume(left_feature, right_feature, max_disp):
    b, c, h, w = left_feature.size()
    cost_volume = left_feature.new_zeros(b, max_disp, h, w)
    for i in range(max_disp):
        if i > 0:
            cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] * right_feature[:, :, :, :-i]).mean(dim=1)
        else:
            cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
    cost_volume = cost_volume.contiguous()
    return cost_volume


def compute_volume(reference_embedding, target_embedding, maxdisp, side='left'):
    batch, channel, height, width = reference_embedding.size()

    cost = torch.zeros(batch, channel, maxdisp, height, width, device='cuda').type_as(reference_embedding)
    cost[:, :, 0, :, :] = reference_embedding - target_embedding
    for idx in range(1, maxdisp):
        if side == 'left':
            cost[:, :, idx, :, idx:] = reference_embedding[:, :, :, idx:] - target_embedding[:, :, :, :-idx]
        if side == 'right':
            cost[:, :, idx, :, :-idx] = target_embedding[:, :, :, idx:] - reference_embedding[:, :, :, :-idx]
    cost = cost.contiguous()

    return cost


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


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W], requires_grad=False)
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def build_corr_volume(img_left, img_right, max_disp):
    B, C, H, W = img_left.shape
    volume = img_left.new_zeros([B, max_disp, H, W])
    for i in range(max_disp):
        if (i > 0) & (i < W):
            volume[:, i, :, i:] = (img_left[:, :, :, i:] * img_right[:, :, :, :W-i]).mean(dim=1)
        else:
            volume[:, i, :, :] = (img_left[:, :, :, :] * img_right[:, :, :, :]).mean(dim=1)

    volume = volume.contiguous()
    return volume


def build_sub_volume(feat_l, feat_r, maxdisp):
    cost = torch.zeros((feat_l.size()[0], maxdisp, feat_l.size()[2], feat_l.size()[3]), device='cuda')
    for i in range(maxdisp):
        cost[:, i, :, :i] = feat_l[:, :, :, :i].abs().sum(1)
        if i > 0:
            cost[:, i, :, i:] = torch.norm(feat_l[:, :, :, i:] - feat_r[:, :, :, :-i], 1, 1)
        else:
            cost[:, i, :, i:] = torch.norm(feat_l[:, :, :, :] - feat_r[:, :, :, :], 1, 1)

    return cost.contiguous()


class InterlacedVolume(nn.Module):
    def __init__(self, num_features=8):
        super(InterlacedVolume, self).__init__()
        self.num_features = num_features

        self.conv3d = nn.Sequential(BasicConv3d(in_channels=1, out_channels=16,
                                                norm_layer=nn.BatchNorm3d, act_layer=nn.ReLU,
                                                kernel_size=(8, 3, 3), stride=(8, 1, 1), padding=(0, 1, 1)),
                                    BasicConv3d(in_channels=16, out_channels=32,
                                                norm_layer=nn.BatchNorm3d, act_layer=nn.ReLU,
                                                kernel_size=(8, 3, 3), stride=(8, 1, 1), padding=(0, 1, 1)),
                                    BasicConv3d(in_channels=32, out_channels=16,
                                                norm_layer=nn.BatchNorm3d, act_layer=nn.ReLU,
                                                kernel_size=(3, 3, 3), stride=(3, 1, 1), padding=(0, 1, 1))
                                    )

        self.volume11 = BasicConv2d(in_channels=16, out_channels=self.num_features,
                                    norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU,
                                    kernel_size=1, stride=1)

    @staticmethod
    def interweave_tensors(refimg_fea, targetimg_fea):
        B, C, H, W = refimg_fea.shape
        interwoven_features = refimg_fea.new_zeros([B, 2 * C, H, W])  # [bz, 192, H/4, W/4]
        interwoven_features[:, ::2, :, :] = refimg_fea
        interwoven_features[:, 1::2, :, :] = targetimg_fea
        interwoven_features = interwoven_features.contiguous()
        return interwoven_features

    def forward(self, feat_l, feat_r, maxdisp):
        B, C, H, W = feat_l.shape
        volume = feat_l.new_zeros([B, self.num_features, maxdisp, H, W])
        for i in range(maxdisp):
            if i > 0:
                x = self.interweave_tensors(feat_l[:, :, :, i:], feat_r[:, :, :, :-i])
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, i:] = x
            else:
                x = self.interweave_tensors(feat_l, feat_r)  # [bz, 192, H/4, W/4]
                x = torch.unsqueeze(x, 1)  # [bz, 1, 192, H/4, W/4]
                x = self.conv3d(x)  # [bz, 16, 1, H/4, W/4]
                x = torch.squeeze(x, 2)  # [bz, 16, H/4, W/4]
                x = self.volume11(x)  # [bz, self.num_features, H/4, W/4]
                volume[:, :, i, :, :] = x

        volume = volume.contiguous()
        return volume
