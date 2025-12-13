# @Time    : 2025/8/20 13:20
# @Author  : zhangchenming
# @Modified: zanregorsek
import torch.nn as nn
import torch.nn.functional as F
import torch
from .debug_utils import debug_printer



class Aggregation(nn.Module):
    def __init__(self, in_channels, left_att, blocks, expanse_ratio, backbone_channels):
        super(Aggregation, self).__init__()
        # blocks = [1, 2, 4] (in yaml file)
        # expanse_ratio = 4 (in yaml file)
        # For input shape [bs, 48, W, H], in_channels = 48

        self.left_att = left_att
        self.expanse_ratio = expanse_ratio

        # conv0: blocks[0] MobileV2Residual layers, no spatial downsampling
        # Input: [bs, 48, W, H] -> Output: [bs, 48, W, H]
        conv0 = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
                 for i in range(blocks[0])] # 0
        self.conv0 = nn.Sequential(*conv0)

        # conv1: Single MobileV2Residual with stride=2, doubles channels, halves spatial dims
        # Input: [bs, 48, W, H] -> Output: [bs, 96, W/2, H/2]
        self.conv1 = MobileV2Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)
        
        # conv2: (blocks[1]-1) MobileV2Residual layers, no spatial/channel changes
        # Input: [bs, 96, W/2, H/2] -> Output: [bs, 96, W/2, H/2]
        conv2_add = [MobileV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)
                     for i in range(blocks[1] - 1)] # 0
        self.conv2 = nn.Sequential(*conv2_add)

        # conv3: Single MobileV2Residual with stride=2, doubles channels, halves spatial dims
        # Input: [bs, 96, W/2, H/2] -> Output: [bs, 192, W/4, H/4]
        self.conv3 = MobileV2Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)
        
        # conv4: (blocks[2]-1) MobileV2Residual layers, no spatial/channel changes
        # Input: [bs, 192, W/4, H/4] -> Output: [bs, 192, W/4, H/4]
        conv4_add = [MobileV2Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)
                     for i in range(blocks[2] - 1)] # 0, 1, 2
        self.conv4 = nn.Sequential(*conv4_add)

        # conv5: Transpose convolution, halves channels, doubles spatial dims
        # Input: [bs, 192, W/4, H/4] -> Output: [bs, 96, W/2, H/2]
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2))

        # conv6: Transpose convolution, halves channels, doubles spatial dims
        # Input: [bs, 96, W/2, H/2] -> Output: [bs, 48, W, H]
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels))

        # Residual connections for skip connections
        # redir1: [bs, 48, W, H] -> [bs, 48, W, H] (for conv6 skip connection)
        # redir2: [bs, 96, W/2, H/2] -> [bs, 96, W/2, H/2] (for conv5 skip connection)
        self.redir1 = MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

        if self.left_att:
            # Attention modules for feature fusion with backbone features
            # att0: processes features at original resolution [bs, 48, W, H]
            # att2: processes features at 1/2 resolution [bs, 96, W/2, H/2]  
            # att4: processes features at 1/4 resolution [bs, 192, W/4, H/4]
            self.att0 = AttentionModule(in_channels, backbone_channels[0])
            self.att2 = AttentionModule(in_channels * 2, backbone_channels[1])
            self.att4 = AttentionModule(in_channels * 4, backbone_channels[2])

    def forward(self, x, features_left):
        # Input x: [bs, 48, W/4, H/4]
        # features[0] W/4 H/4
        # features[1] W/8 H/8
        # features[2] W/16 H/16
        # features[3] W/32 H/32
        
        # print(f"aggegation: type of x: {x.dtype}")
        # print(f"aggregation: conv0 weights type: {self.conv0[0].pwconv[0].weight.dtype}")
        # print(f"aggegation: type of x: {x.type()}")
        # print(f"aggregation: conv0 weights type: {self.conv0[0].pwconv[0].weight.type()}")

        # First stage: Apply conv0 blocks, maintain dimensions
        # x: [bs, 48, W/4, H/4] -> [bs, 48, W/4, H/4]
        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"Aggregation: features_left[0] Found NaN!", force_print=features_left[0].isfinite().all()==False)
            debug_printer.print_of_function_force_print(lambda : f"input x Found NaN!", force_print=x.isfinite().all()==False)

        x = self.conv0(x)

        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"conv0 x Found NaN!", force_print=x.isfinite().all()==False)

        if self.left_att:
            # Apply attention with backbone features at same resolution
            # x: [bs, 48, W, H], features_left[0]: [bs, backbone_channels[0], W, H]
            # Output: [bs, 48, W/4, H/4]
            x = self.att0(x, features_left[0])

        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"att0 Found NaN!", force_print=x.isfinite().all()==False)

        # Second stage: Downsample and increase channels
        # conv1: [bs, 48, W/4, H/4] -> [bs, 96, W/8, H/8]
        conv1 = self.conv1(x)
        # conv2: [bs, 96, W/8, H/8] -> [bs, 96, W/8, H/8] (no change)
        conv2 = self.conv2(conv1)
        if self.left_att:
            # Apply attention with backbone features at 1/2 resolution
            # conv2: [bs, 96, W/2, H/2], features_left[1]: [bs, backbone_channels[1], W/2, H/2]
            # Output: [bs, 96, W/8, H/8]
            conv2 = self.att2(conv2, features_left[1]) # I erronously got: cost 1/16, x 1/8

        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"att2 Found NaN!", force_print=conv2.isfinite().all()==False)

        # Third stage: Further downsample and increase channels
        # conv3: [bs, 96, W/8, H/8] -> [bs, 192, W/16, H/16]
        conv3 = self.conv3(conv2)
        # conv4: [bs, 192, W/16, H/16] -> [bs, 192, W/16, H/16] (no change)
        conv4 = self.conv4(conv3)
        if self.left_att:
            # Apply attention with backbone features at 1/4 resolution
            # conv4: [bs, 192, W/4, H/4], features_left[2]: [bs, backbone_channels[2], W/4, H/4]
            # Output: [bs, 192, W/16, H/16]
            conv4 = self.att4(conv4, features_left[2])

        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"att4 Found NaN!", force_print=conv4.isfinite().all()==False)

        # Decoder stage: Upsample back to original resolution with skip connections
        # conv5: [bs, 192, W/4, H/4] -> [bs, 96, W/2, H/2]
        # redir2(conv2): [bs, 96, W/2, H/2] -> [bs, 96, W/2, H/2]
        # Element-wise addition: [bs, 96, W/2, H/2] + [bs, 96, W/2, H/2] = [bs, 96, W/2, H/2]
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        
        # conv6: [bs, 96, W/2, H/2] -> [bs, 48, W, H]
        # redir1(x): [bs, 48, W, H] -> [bs, 48, W, H]
        # Element-wise addition: [bs, 48, W, H] + [bs, 48, W, H] = [bs, 48, W, H]
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        # Final output: [bs, 48, W/4, H/4] (same as input dimensions)
        return [conv6]


class MobileV2Residual(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2Residual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation

        # v2
        self.pwconv = nn.Sequential(
            # pointwise: expands along channel dimension
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.dwconv = nn.Sequential(
            # depthwise: spatial evaluation, no cross channel mixing due to groups=hidden_dim
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.pwliner = nn.Sequential(
            # pointwise linear convolution
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        # v2
        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"MobileV2Residual input x Found NaN!", force_print=x.isfinite().all()==False)
        feat = self.pwconv(x)
        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"MobileV2Residual after pwconv Found NaN!", force_print=feat.isfinite().all()==False)
        feat = self.dwconv(feat)
        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"MobileV2Residual after dwconv Found NaN!", force_print=feat.isfinite().all()==False)
        feat = self.pwliner(feat)
        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"MobileV2Residual after pwliner Found NaN!", force_print=feat.isfinite().all()==False)

        if self.use_res_connect:
            return x + feat
        else:
            return feat


class AttentionModule(nn.Module):
    def __init__(self, dim, img_feat_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(img_feat_dim, dim, 1)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, cost, x): # I erronously got: cost 1/16, x 1/8
        attn = self.conv0(x)

        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"attn Found NaN!", force_print=attn.isfinite().all()==False)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"attn_0 Found NaN!", force_print=attn_0.isfinite().all()==False)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"attn_1 Found NaN!", force_print=attn_1.isfinite().all()==False)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"att2 Found NaN!", force_print=attn_2.isfinite().all()==False)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)

        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"attn final Found NaN!", force_print=attn.isfinite().all()==False)

        return attn * cost # RuntimeError: The size of tensor a (92) must match the size of tensor b (46) at non-singleton dimension 3
        # The size of a 1/8 must match the size of b 1/16    

class BasicDepthEnrichment(nn.Module):
    def __init__(self, intermediate_channels=[48, 32, 16], out_depth_bins=[48, 48, 24], mlp_hidden_dim=128, convs_dilated=True):
        """
        BasicDepthEnrichment processes 2D depth maps with multi-scale convolutions.
        
        Args:
            intermediate_channels (list): Intermediate channels for conv filters [3x3, 5x5, 7x7]. Default: [48, 32, 16]
            out_depth_bins (list): Final output channel sizes for each scale. Default: [48, 48, 24]
            mlp_hidden_dim (int): Hidden dimension for MLP layers. Default: 128
            convs_dilated (bool): Use dilated convolutions (True) or classic convolutions (False). Default: True
        """
        super(BasicDepthEnrichment, self).__init__()
        
        self.intermediate_channels = intermediate_channels
        self.out_depth_bins = out_depth_bins
        self.convs_dilated = convs_dilated
        total_channels = sum(intermediate_channels)  # 48 + 32 + 16 = 96
        
        # Multi-scale convolutions with different kernel sizes
        # All use input_channels=1 (single depth channel)
        if convs_dilated:
            # Dilated convolutions: use 3x3 kernel with different dilation rates
            self.conv_3x3 = nn.Conv2d(
                in_channels=1, 
                out_channels=intermediate_channels[0],  # 48
                kernel_size=3, 
                padding=1,  # dilation=1, padding=1
                dilation=1,
                bias=False
            )
            
            self.conv_5x5 = nn.Conv2d(
                in_channels=1, 
                out_channels=intermediate_channels[1],  # 32
                kernel_size=3, 
                padding=2,  # dilation=2, padding=2 for same spatial size
                dilation=2,
                bias=False
            )
            
            self.conv_7x7 = nn.Conv2d(
                in_channels=1, 
                out_channels=intermediate_channels[2],  # 16
                kernel_size=3, 
                padding=3,  # dilation=3, padding=3 for same spatial size
                dilation=3,
                bias=False
            )
        else:
            # Classic convolutions: use different kernel sizes
            self.conv_3x3 = nn.Conv2d(
                in_channels=1, 
                out_channels=intermediate_channels[0],  # 48
                kernel_size=3, 
                padding=1,  # Same spatial size
                bias=False
            )
            
            self.conv_5x5 = nn.Conv2d(
                in_channels=1, 
                out_channels=intermediate_channels[1],  # 32
                kernel_size=5, 
                padding=2,  # Same spatial size
                bias=False
            )
            
            self.conv_7x7 = nn.Conv2d(
                in_channels=1, 
                out_channels=intermediate_channels[2],  # 16
                kernel_size=7, 
                padding=3,  # Same spatial size
                bias=False
            )
        
        # Batch normalization for each conv
        self.bn_3x3 = nn.BatchNorm2d(intermediate_channels[0])
        self.bn_5x5 = nn.BatchNorm2d(intermediate_channels[1])
        self.bn_7x7 = nn.BatchNorm2d(intermediate_channels[2])
        
        # First MLP: reduces concatenated channels to out_depth_bins[0]
        # Input: total_channels (96) -> Hidden -> Output: out_depth_bins[0] (48)
        self.mlp1 = nn.Sequential(
            nn.Conv2d(total_channels, mlp_hidden_dim, 1, bias=False),
            nn.BatchNorm2d(mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_hidden_dim, out_depth_bins[0], 1, bias=False),
            nn.BatchNorm2d(out_depth_bins[0]),
            # nn.GroupNorm(1, out_depth_bins[0]),  # GroupNorm with 1 group = LayerNorm
            # nn.Hardtanh(min_val=-3.0, max_val=3.0, inplace=True)  # Fast bounded activation
            nn.ReLU(inplace=True),
        )
        
        # Additional conv2d after mlp1
        # Input: out_depth_bins[0] (48) -> Output: 96 channels
        self.additional_conv = nn.Conv2d(
            in_channels=out_depth_bins[0],
            out_channels=96,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn_additional = nn.BatchNorm2d(96)
        
        # Additional MLP after conv2d
        # Input: 96 -> Hidden -> Output: out_depth_bins[0] (48)
        self.mlp_additional = nn.Sequential(
            nn.Conv2d(96, mlp_hidden_dim, 1, bias=False),
            nn.BatchNorm2d(mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_hidden_dim, out_depth_bins[0], 1, bias=True),
            nn.BatchNorm2d(out_depth_bins[0]),
            nn.ReLU(inplace=True),
            # nn.GroupNorm(1, out_depth_bins[0]),  # GroupNorm with 1 group = LayerNorm
            # nn.Hardtanh(min_val=-3.0, max_val=3.0, inplace=True)  # Fast bounded activation
        )
        
        # First downsampling convolution (stride=2)
        # Input: out_depth_bins[0] (48) -> Output: 96 channels, spatial size halved
        self.downsample_conv1 = nn.Conv2d(
            in_channels=out_depth_bins[0],
            out_channels=96,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn_downsample1 = nn.BatchNorm2d(96)
        
        # Second MLP: reduces downsampled features to out_depth_bins[1]
        # Input: 96 -> Hidden -> Output: out_depth_bins[1] (48)
        self.mlp2 = nn.Sequential(
            nn.Conv2d(96, mlp_hidden_dim, 1, bias=False),
            nn.BatchNorm2d(mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_hidden_dim, out_depth_bins[1], 1, bias=True),
            nn.BatchNorm2d(out_depth_bins[1]),
            nn.ReLU(inplace=True),
            # nn.GroupNorm(1, out_depth_bins[1]),  # GroupNorm with 1 group = LayerNorm
            # nn.Hardtanh(min_val=-3.0, max_val=3.0, inplace=True)  # Fast bounded activation
        )
        
        # Second downsampling convolution (stride=2)
        # Input: out_depth_bins[1] (48) -> Output: 64 channels, spatial size halved again
        self.downsample_conv2 = nn.Conv2d(
            in_channels=out_depth_bins[1],
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn_downsample2 = nn.BatchNorm2d(64)
        
        # Third MLP: reduces second downsampled features to out_depth_bins[2]
        # Input: 64 -> Hidden -> Output: out_depth_bins[2] (24)
        self.mlp3 = nn.Sequential(
            nn.Conv2d(64, mlp_hidden_dim, 1, bias=False),
            nn.BatchNorm2d(mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_hidden_dim, out_depth_bins[2], 1, bias=True),
            nn.BatchNorm2d(out_depth_bins[2]),
            nn.ReLU(inplace=True),
            # nn.GroupNorm(1, out_depth_bins[2]),  # GroupNorm with 1 group = LayerNorm
            # nn.Hardtanh(min_val=-3.0, max_val=3.0, inplace=True)  # Fast bounded activation
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch, width, height]
            
        Returns:
            List of three outputs:
            - output2: [batch, out_depth_bins[1], width/2, height/2] 
            - output3: [batch, out_depth_bins[2], width/4, height/4]
        """
        # Add channel dimension: [batch, width, height] -> [batch, 1, width, height]
        # print(f"Shape before unsqueeze: {x.shape}")
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # print(f"Shape after unsqueeze: {x.shape}")

        batch, channels, width, height = x.shape
        assert channels == 1, f"Expected 1 input channel, got {channels}"

        # print(f"enrichment: type of x: {x.dtype}")
        # print(f"enrichment: conv_3x3 weights type: {self.conv_3x3.weight.dtype}")
        # print(f"enrichment: type of x: {x.type()}")
        # print(f"enrichment: conv_3x3 weights type: {self.conv_3x3.weight.type()}")
        # print(f"enrichment: bn_3x3 weight type: {self.bn_3x3.weight.type()}")
        # print(f"enrichment: bn_3x3 bias type: {self.bn_3x3.bias.type()}")
        
        # Multi-scale convolutions with same spatial dimensions
        # conv_3x3: [batch, 1, width, height] -> [batch, 48, width, height]
        # feat_3x3 = F.relu(self.bn_3x3(self.conv_3x3(x)), inplace=True)
        feat_3x3 = F.relu6(self.bn_3x3(self.conv_3x3(x)), inplace=True)
        
        # conv_5x5: [batch, 1, width, height] -> [batch, 32, width, height]  
        # feat_5x5 = F.relu(self.bn_5x5(self.conv_5x5(x)), inplace=True)
        feat_5x5 = F.relu6(self.bn_5x5(self.conv_5x5(x)), inplace=True)
        
        # conv_7x7: [batch, 1, width, height] -> [batch, 16, width, height]
        # feat_7x7 = F.relu(self.bn_7x7(self.conv_7x7(x)), inplace=True)
        feat_7x7 = F.relu6(self.bn_7x7(self.conv_7x7(x)), inplace=True)
        
        # Concatenate along channel dimension
        # [batch, 48+32+16, width, height] = [batch, 96, width, height]
        concat_feats = torch.cat([feat_3x3, feat_5x5, feat_7x7], dim=1)
        
        # First output: Apply MLP to reduce channels
        # [batch, 96, width, height] -> [batch, out_depth_bins[0], width, height]
        output1 = self.mlp1(concat_feats)  # [batch, 48, width, height]
        
        # Additional conv2d processing
        # [batch, 48, width, height] -> [batch, 96, width, height]
        # additional_conv_out = F.relu(self.bn_additional(self.additional_conv(output1)), inplace=True)
        additional_conv_out = F.relu(self.bn_additional(self.additional_conv(output1)), inplace=True)
        
        # Additional MLP processing  
        # [batch, 96, width, height] -> [batch, 48, width, height]
        additional_mlp_out = self.mlp_additional(additional_conv_out)
        
        # First downsampling convolution with stride=2
        # [batch, 48, width, height] -> [batch, 96, width/2, height/2]
        downsampled1 = F.relu(self.bn_downsample1(self.downsample_conv1(additional_mlp_out)), inplace=True)
        
        # Second output: Apply MLP to reduce channels  
        # [batch, 96, width/2, height/2] -> [batch, out_depth_bins[1], width/2, height/2]
        output2 = self.mlp2(downsampled1)  # [batch, 48, width/2, height/2]
        
        # Second downsampling convolution with stride=2
        # [batch, 48, width/2, height/2] -> [batch, 64, width/4, height/4]
        downsampled2 = F.relu(self.bn_downsample2(self.downsample_conv2(output2)), inplace=True)
        
        # Third output: Apply MLP to reduce channels
        # [batch, 64, width/4, height/4] -> [batch, out_depth_bins[2], width/4, height/4]
        output3 = self.mlp3(downsampled2)  # [batch, 24, width/4, height/4]
        
        return [output1, output2, output3]
    

class AggregationLittle(nn.Module):
    def __init__(self, depth_channels_in, left_att, blocks, expanse_ratio, backbone_channels):
        super(AggregationLittle, self).__init__()
        # blocks = [1, 2] (only first two values used)
        # expanse_ratio = 4 (in yaml file)
        # For input shape [bs, 48, W, H], depth_channels_in = 48

        # depth_channels_in = 48, 24
        # backbone_channels = # [16, 24, 40, 96] # mobileNetV3_small

        self.left_att = left_att
        self.expanse_ratio = expanse_ratio

        # conv0: blocks[0] MobileV2Residual layers, no spatial downsampling
        # Input: [bs, 48, W, H] -> Output: [bs, 48, W, H]
        conv0 = [MobileV2Residual(depth_channels_in[0], depth_channels_in[0], stride=1, expanse_ratio=self.expanse_ratio)
                 for i in range(blocks[0])]
        self.conv0 = nn.Sequential(*conv0)

        # conv1: Single MobileV2Residual with stride=2, doubles channels, halves spatial dims
        # Input: [bs, 48, W, H] -> Output: [bs, 96, W/2, H/2]
        self.conv1 = MobileV2Residual(depth_channels_in[0], depth_channels_in[0] * 2, stride=2, expanse_ratio=self.expanse_ratio)

        ch_inter = (depth_channels_in[0] * 2 + depth_channels_in[1]) // self.expanse_ratio

        self.channel_interconnect = nn.Sequential(
            nn.Conv2d(depth_channels_in[0] * 2 + depth_channels_in[1], ch_inter, kernel_size=1, bias=False),  # 128 -> 96
            nn.BatchNorm2d(ch_inter),
            nn.ReLU(inplace=True)
        )
        
        # conv2: (blocks[1]-1) MobileV2Residual layers, no spatial/channel changes
        # Input: [bs, 96, W/2, H/2] -> Output: [bs, 96, W/2, H/2]
        conv2_add = [MobileV2Residual(ch_inter, ch_inter, stride=1, expanse_ratio=self.expanse_ratio)
                     for i in range(blocks[1] - 1)]
        self.conv2 = nn.Sequential(*conv2_add)

        # conv3: Transpose convolution, halves channels, doubles spatial dims
        # Input: [bs, 96, W/2, H/2] -> Output: [bs, 48, W, H]
        # ch_conv3 = ch_inter * 2
        ch_conv3 = depth_channels_in[0] # the same as input
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(ch_inter, ch_conv3, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(ch_conv3)
        )

        # ch_conv4 = depth_channels_in[0] # the same as input
        # self.conv4 = nn.Sequential(
        #     nn.ConvTranspose2d(ch_inter, ch_conv4, 3, padding=1, output_padding=1, stride=2, bias=False),
        #     nn.BatchNorm2d(ch_conv4)
        # )

        # Residual connection for skip connection
        # redir1: [bs, 48, W, H] -> [bs, 48, W, H] (for conv3 skip connection)
        self.redir1 = MobileV2Residual(depth_channels_in[0], ch_conv3, stride=1, expanse_ratio=self.expanse_ratio)
        # self.redir2 = MobileV2Residual(ch_inter, backbone_channels[0], stride=1, expanse_ratio=self.expanse_ratio)

        if self.left_att:
            # Attention modules for feature fusion with backbone features
            # att0: processes features at original resolution [bs, 48, W, H]
            # att2: processes features at 1/2 resolution [bs, 96, W/2, H/2]  
            self.att0 = AttentionModule(depth_channels_in[0], backbone_channels[0])
            self.att2 = AttentionModule(ch_inter, backbone_channels[1])

    def forward(self, depth, features_left):

        # depth channels -> [self.max_disp // 4, self.max_disp // 8] = 48, 24
        # depth[0] = W/4 H/4
        # depth[1] = W/8 H/8

        # features[0] W/4 H/4
        # features[1] W/8 H/8
        # features[2] W/16 H/16
        # features[3] W/32 H/32
        
        # for i, dp in enumerate(depth):
        #     print(f"depth[{i}] shape: {dp.shape}")
        # for i, fl in enumerate(features_left):
        #     print(f"features_left[{i}] shape: {fl.shape}")

        x = depth[0] 

        # First stage: Apply conv0 blocks, maintain dimensions
        # x: [bs, 48, W/4, H/4] -> [bs, 48, W/4, H/4]
        conv0 = F.relu6(self.conv0(x)) # 48 ch
        # print(f"aggregation_little: shape of x = self.conv0(x) = {conv0.shape}")
        if self.left_att:
            # Apply attention with backbone features at same resolution
            # x: [bs, 48, W, H], features_left[0]: [bs, backbone_channels[0], W, H]
            # Output: [bs, 48, W/4, H/4]
            att0 = self.att0(conv0, features_left[0]) #48 channels

        # Second stage: Downsample and increase channels
        # conv1: [bs, 48, W/4, H/4] -> [bs, 96, W/8, H/8]
        conv1 = F.relu6(self.conv1(att0)) # 96 ch
        cat1 = torch.cat([depth[1], conv1 ], dim=1) # depth_channels[1] + depth_channels_in[0] * 2 = 24 + 96 = 120
        # print(f"shape conv1: {conv1.shape}")
        # print(f"shape depth[1]: {depth[1].shape}")
        # print(f"shape cat1: {cat1.shape}")
        cat1 = self.channel_interconnect(cat1) # ch_inter

        
        # conv2: [bs, 96, W/8, H/8] -> [bs, 96, W/8, H/8] (no change)
        conv2 = F.relu6(self.conv2(cat1)) #
        # print(f"shape conv2: {conv2.shape}")
        
        if self.left_att:
            # Apply attention with backbone features at 1/2 resolution
            # Output: [bs, 96, W/8, H/8]
            att2 = self.att2(conv2, features_left[1]) # ch_inter
        # print(f"shape conv2+att2: {conv2.shape}")

        # option A
        a = F.relu6(self.conv3(att2)) # up samples to W/4 W/4
        # a = conv2
        b = F.relu6(self.redir1(att0)) # is [bs, 48, W/4, H/4]
        # print(f"shape a: conv2: {a.shape}")
        # print(f"shape b: redir2: {b.shape}")
        # conv3 = F.relu(self.conv3(conv2) + self.redir2(conv2), inplace=True)

        from .debug_utils import debug_printer

        debug_printer.print_of_function(lambda : f"a stats: min={a.min():.3f}, max={a.max():.3f}, mean={a.mean():.3f}, std={a.std():.3f}")
        debug_printer.print_of_function(lambda : f"b stats: min={b.min():.3f}, max={b.max():.3f}, mean={b.mean():.3f}, std={b.std():.3f}")

        conv_out = F.relu(a + b, inplace=True)
        # conv_out =  nn.Hardtanh(min_val=-3.0, max_val=3.0, inplace=True)(a + b)

        # conv_out = F.relu(self.conv3(att2) + self.redir2(att0), inplace=True)
        
        # print(f"shape conv_out: {conv_out.shape}")

        # or option B
        # a = self.conv3(att0)
        # b = self.redir2(conv2)
        # conv4 = F.relu(self.conv3(conv2) + self.redir2(conv2), inplace=True)
        
        # Decoder stage: Upsample back to original resolution with skip connection
        # conv3: [bs, 96, W/2, H/2] -> [bs, 48, W, H]
        # redir1(x): [bs, 48, W, H] -> [bs, 48, W, H]
        # Element-wise addition: [bs, 48, W/4, H/4] + [bs, 48, W/4, H/4] = [bs, 48, W/4, H/4]

        # conv4 = F.relu(self.conv4(conv3) + self.redir1(conv0), inplace=True)
        # Final output: [bs, 48, W/4, H/4] (same as input dimensions)
        # return [conv4]
        return [conv_out]

class Efficient3DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), stride=1, padding=1):
        super().__init__()
        kd, kh, kw = kernel_size
        
        # 1. Depthwise separable for spatial dimensions (H,W)
        self.spatial_depthwise = nn.Conv3d(
            in_channels, in_channels, 
            kernel_size=(1, kh, kw),  # Only spatial
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            groups=in_channels,  # Depthwise
            bias=False
        )
        
        # 2. 1D convolution along depth/disparity dimension
        self.depth_conv = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=(kd, 1, 1),  # Only depth
            stride=(stride, 1, 1),
            padding=(padding, 0, 0),
            groups=in_channels,  # Depthwise
            bias=False
        )
        
        # 3. Pointwise (1x1x1) to mix channels
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.bn3 = nn.BatchNorm3d(out_channels)
        
    def forward(self, x):
        # Process spatial dimensions
        x = F.relu(self.bn1(self.spatial_depthwise(x)))
        # Process depth dimension
        x = F.relu(self.bn2(self.depth_conv(x)))
        # Mix channels
        x = self.bn3(self.pointwise(x))
        return x
    
# from article: https://sh-tsang.medium.com/paper-p3d-pseudo-3d-residual-networks-video-classification-action-recognition-d1dd13638d7c
class Pseudo3DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3)):
        super().__init__()
        kd, kh, kw = kernel_size
        mid_channels = in_channels
        
        # 2D spatial convolution (no depth)
        self.spatial_conv = nn.Conv3d(
            in_channels, mid_channels,
            kernel_size=(1, kh, kw),
            padding=(0, kh//2, kw//2),
            bias=False
        )
        
        # 1D depth convolution
        self.temporal_conv = nn.Conv3d(
            mid_channels, out_channels,
            kernel_size=(kd, 1, 1),
            padding=(kd//2, 0, 0),
            bias=False
        )
        
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.spatial_conv(x)))
        x = self.bn2(self.temporal_conv(x))
        return x