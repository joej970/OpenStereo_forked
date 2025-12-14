# @Time    : 2025/8/20 13:20
# @Author: zanregorsek
import torch.nn as nn
import torch
from torch.nn import functional as F

from .aggregation import MobileV2Residual

class FuseDepth(nn.Module):
    def __init__(self, fusion_type, src_0_depth_bins, src_1_depth_bins, out_depth_bins, num_convs = 4, expanse_ratio=4, ignore_add_depth_debug_only=False):
        """
        FuseDepth fuses two depth feature maps with different fusion strategies.
        
        Args:
            fusion_type (str): Type of fusion - "concat", "addition", or "multiplication"
            src_0_depth_bins (int): Number of depth bins in first input
            src_1_depth_bins (int): Number of depth bins in second input  
            out_depth_bins (int): Number of output depth bins
            num_convs (int): Number of MobileV2Residual layers to apply after fusion
            expanse_ratio (int): Expansion ratio for MobileV2Residual layers. Default: 4
        """
        super(FuseDepth, self).__init__()

        self.input_normalization_0 = nn.BatchNorm2d(src_0_depth_bins)
        self.input_normalization_1 = nn.BatchNorm2d(src_1_depth_bins)

        self.fusion_type = fusion_type
        self.src_0_depth_bins = src_0_depth_bins
        self.src_1_depth_bins = src_1_depth_bins
        self.out_depth_bins = out_depth_bins
        self.num_convs = num_convs
        self.expanse_ratio = expanse_ratio

        self.ignore_add_depth_debug_only = ignore_add_depth_debug_only

        if (self.ignore_add_depth_debug_only):
            print(f"Second source will be ignored.")
        
        # Validate fusion type
        assert fusion_type in ["concat", "addition", "multiplication"], \
            f"fusion_type must be one of ['concat', 'addition', 'multiplication'], got {fusion_type}"
        
        # Determine input channels after fusion
        if fusion_type == "concat":
            
            # 3D convolution to process concatenated inputs along new dimension
            self.conv3d_fusion = nn.Sequential(
                nn.Conv3d(2, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(1),
                nn.ReLU6(inplace=True)
            )
            # Channel adjustment layer to match output dimensions  
            if src_0_depth_bins != out_depth_bins:
                self.channel_mixing = nn.Sequential(
                    nn.Conv2d(src_0_depth_bins, out_depth_bins, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_depth_bins),
                    nn.ReLU(inplace=True)
                )
            else:
                self.channel_mixing = nn.Identity()
        elif fusion_type in ["addition", "multiplication"]:
            # For addition and multiplication, inputs must have same number of channels
            assert src_0_depth_bins == src_1_depth_bins, \
                f"For {fusion_type} fusion, src_0_depth_bins ({src_0_depth_bins}) must equal src_1_depth_bins ({src_1_depth_bins})"
            
            self.channel_mixing = nn.Identity()  # No channel adjustment needed
        
                
        # MobileV2Residual layers for post-fusion processing
        if num_convs > 0:
            conv_layers = [
                MobileV2Residual(out_depth_bins, out_depth_bins, stride=1, expanse_ratio=expanse_ratio)
                for _ in range(num_convs)
            ]
            # add activation ReLU6
            conv_layers.append(nn.ReLU6(inplace=True))
            self.conv_layers = nn.Sequential(*conv_layers)
        else:
            self.conv_layers = nn.Identity()

        # self.layer_norm = nn.GroupNorm(1, src_0_depth_bins)
        self.layer_norm = nn.BatchNorm2d(src_0_depth_bins)
    
    def forward(self, src_0, src_1):
        """
        Forward pass for depth fusion.
        
        Args:
            src_0: First input tensor of shape [batch, src_0_depth_bins, width, height]
            src_1: Second input tensor of shape [batch, src_1_depth_bins, width, height]
            
        Returns:
            Fused tensor of shape [batch, out_depth_bins, width, height]
        """
        batch, c0, height, width = src_0.shape
        batch1, c1, height1, width1 = src_1.shape
        
        # Validate input dimensions
        assert batch == batch1 and height == height1 and width == width1, \
            f"Spatial dimensions must match: src_0 {src_0.shape} vs src_1 {src_1.shape}"
        assert c0 == self.src_0_depth_bins, \
            f"src_0 channels ({c0}) must match src_0_depth_bins ({self.src_0_depth_bins})"
        assert c1 == self.src_1_depth_bins, \
            f"src_1 channels ({c1}) must match src_1_depth_bins ({self.src_1_depth_bins})"
        
        src_0 = self.input_normalization_0(src_0)
        src_1 = self.input_normalization_1(src_1)

        if self.ignore_add_depth_debug_only:
            fused = src_0
        else:
            # Apply fusion strategy
            if self.fusion_type == "concat":
                # Add extra dimension: [batch, depth_bins, height, width] -> [batch, 1, depth_bins, height, width]
                src_0_expanded = src_0.unsqueeze(1)  # [batch, 1, src_0_depth_bins, height, width]
                src_1_expanded = src_1.unsqueeze(1)  # [batch, 1, src_1_depth_bins, height, width]
                
                # Concatenate along new dimension: [batch, 2, depth_bins, height, width]
                # Note: This requires both inputs to have same depth_bins for concatenation
                assert c0 == c1, f"For 3D concat fusion, both inputs must have same depth_bins: {c0} vs {c1}"
                fused_3d = torch.cat([src_0_expanded, src_1_expanded], dim=1)  # [batch, 2, depth_bins, height, width]
                
                # Apply 3D convolution to reduce back to 4D: [batch, 2, depth_bins, height, width] -> [batch, 1, depth_bins, height, width]
                conv3d_out = self.conv3d_fusion(fused_3d)  # [batch, 1, depth_bins, height, width]
                
                # Remove extra dimension: [batch, 1, depth_bins, height, width] -> [batch, depth_bins, height, width]
                fused = conv3d_out.squeeze(1)  # [batch, depth_bins, height, width]
                
            elif self.fusion_type == "addition":
                # Element-wise addition
                # [batch, src_0_depth_bins, height, width] (same as input)
                fused = src_0 + src_1
                
            elif self.fusion_type == "multiplication":
                # Element-wise multiplication
                # [batch, src_0_depth_bins, height, width] (same as input)
                # print(f"fusion.py: multiplication: src_0: {src_0.shape}, src_1: {src_1.shape}")
                # print(f"fusion.py: multiplication: src_0 max, min, contains nan: {torch.max(src_0)}, {torch.min(src_0)}, {torch.isnan(src_0).any()}")
                # print(f"fusion.py: multiplication: src_1 max, min, contains nan: {torch.max(src_1)}, {torch.min(src_1)}, {torch.isnan(src_1).any()}")
                
                fused = src_0 * src_1
        
        # fused = F.layer_norm(fused, fused.shape[1:])
        fused = self.layer_norm(fused)

        # Adjust channels to match output dimensions
        # [batch, fused_channels, height, width] -> [batch, out_depth_bins, height, width]
        adjusted = self.channel_mixing(fused)
        
        # Apply MobileV2Residual layers for further processing
        # [batch, out_depth_bins, height, width] -> [batch, out_depth_bins, height, width]
        output = self.conv_layers(adjusted)
        
        return [output]
    
