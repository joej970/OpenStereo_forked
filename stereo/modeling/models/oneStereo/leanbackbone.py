import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.debug_utils import debug_printer

__all__ = ['FeatureExtractionNet']

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, conv="2D"):
        super(ConvBNReLU, self).__init__()
        if conv=="2D":
            self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
            self.bn = nn.BatchNorm2d(out_chan)
        else:
            self.conv = nn.Conv3d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
            self.bn = nn.BatchNorm3d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat

class ShallowBranch(nn.Module):

    def __init__(self, out_channels=128):
        self.out_channels = out_channels
        super(ShallowBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2), # W/2, H/2
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2), # W/4, H/4
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, out_channels, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(out_channels, 128, 3, stride=2), # W/8, H/8
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, out_channels, 3, stride=1),
        )

    def forward(self, x):
        feat_2 = self.S1(x) # W/2, H/2
        feat_4 = self.S2(feat_2) # W/4, H/4
        feat_8 = self.S3(feat_4) # W/8, H/8
        return [feat_2, feat_4, feat_8] # [W/2, H/2 | W/4, H/4 | W/8, H/8]

class StemBlock(nn.Module):

    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x) # 1/2
        feat_left = self.left(feat) # 1/4
        feat_right = self.right(feat) # 1/4
        feat = torch.cat([feat_left, feat_right], dim=1) # 1/4
        feat = self.fuse(feat) # 1/4
        return feat

class CEBlock(nn.Module):

    def __init__(self, channels=128):
        super(CEBlock, self).__init__()
        self.channels = channels
        self.bn = nn.BatchNorm2d(channels)
        self.conv_gap = ConvBNReLU(channels, channels, 1, stride=1, padding=0)
        # TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(channels, channels, 3, stride=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat

class GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat # W -> W

class GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        ) # /2
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_chan, in_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.Conv2d(
                in_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat # W -> W/2

class DeepBranch(nn.Module):

    # GE: Gather and Expansion
    def __init__(self, channels=128):
        super(DeepBranch, self).__init__()

        self.channels = channels
        self.S1S2 = StemBlock() # S1= 1/2 S2= 1/4
        self.S3 = nn.Sequential( 
            GELayerS2(16, 32, exp_ratio=6), # default exp_ratio=6
            GELayerS1(32, 32, exp_ratio=6),
        ) # S3= 1/8
        self.S4 = nn.Sequential(
            GELayerS2(32, 64, exp_ratio=6),
            GELayerS1(64, 64, exp_ratio=6),
        ) # S4= 1/16
        self.S5_4 = nn.Sequential(
            GELayerS2(64, self.channels, exp_ratio=3),
            GELayerS1(self.channels, self.channels, exp_ratio=3),
            GELayerS1(self.channels, self.channels, exp_ratio=3),
            GELayerS1(self.channels, self.channels, exp_ratio=3),
        ) # S5_4= 1/32
        self.S5_5 = CEBlock(self.channels) # no change in size

    def forward(self, x):
        feat2 = self.S1S2(x)  # S1= 1/2 S2= 1/4
        feat3 = self.S3(feat2)  # S3= 1/8
        feat4 = self.S4(feat3)  # S4= 1/16
        feat5_4 = self.S5_4(feat4)  # S5_4= 1/32
        feat5_5 = self.S5_5(feat5_4) # 1/32
        return feat2, feat3, feat4, feat5_4, feat5_5 # [W/4, H/4 | W/8, H/8 | W/16, H/16 | W/32, H/32 | W/32, H/32]
    
class AggregationLayer(nn.Module):

    def __init__(self, channels = 128):
        super(AggregationLayer, self).__init__()

        self.channels = channels

        self.left1 = nn.Sequential(
            nn.Conv2d(
                self.channels, self.channels, kernel_size=3, stride=1,
                padding=1, groups=self.channels, bias=False), # H, W
            nn.BatchNorm2d(self.channels),
            nn.Conv2d(
                self.channels, self.channels, kernel_size=1, stride=1,
                padding=0, bias=False), # H, W
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                self.channels, self.channels, kernel_size=3, stride=2,
                padding=1, bias=False), # H/2, W/2
            nn.BatchNorm2d(self.channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False) # H/4, W/4
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                self.channels, self.channels, kernel_size=3, stride=1,
                padding=1, bias=False), # H, W
            nn.BatchNorm2d(self.channels),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                self.channels, self.channels, kernel_size=3, stride=1,
                padding=1, groups=self.channels, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.Conv2d(
                self.channels, self.channels, kernel_size=1, stride=1,
                padding=0, bias=False), # H, W
        )
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                self.channels, self.channels, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),  # not shown in paper
        )

        self.upconv_first = nn.Sequential(
            nn.ConvTranspose2d(
                self.channels, self.channels, kernel_size=2, stride=2,
                padding=0, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                self.channels, self.channels, kernel_size=2, stride=2,
                padding=0, bias=False),
            nn.BatchNorm2d(self.channels))
        
        self.upconv_second = nn.Sequential(
            nn.ConvTranspose2d(
                self.channels, self.channels, kernel_size=2, stride=2,
                padding=0, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                self.channels, self.channels, kernel_size=2, stride=2,
                padding=0, bias=False),
            nn.BatchNorm2d(self.channels))


    def forward(self, x_d, x_s): # [W/8, H/8], [W/32, H/32] -> [W/8, H/8]
        # print(f"Aggregation layer input sizes: x_d: {x_d.size()}, x_s: {x_s.size()}")
        # x_d: torch.Size([16, 128, 40, 92]), x_s: torch.Size([16, 128, 10, 23])
        dsize = x_d.size()[2:] # 40, 92 = H/8, W/8
        detail1 = self.left1(x_d) # Con2d + BN + Conv2d: H/8, W/8
        detail2 = self.left2(x_d) # Conv2d + BN + AvgPool: H/32, W/32
        context1 = self.right1(x_s) # Conv2d + BN: H/32, W/32
        context2 = self.right2(x_s) # Conv2d + BN + Conv2d: H/32, W/32
        context1 = self.upconv_first(context1) # H/32, W/32 -> H/8, W/8
        # context1 = F.interpolate(
        #     context1, scale_factor=4, mode='bilinear', align_corners=False) # H/32, W/32 -> H/8, W/8
        # context1 = F.interpolate(
        #     context1, size=dsize, mode='bilinear', align_corners=True) # H/32, W/32 -> H/8, W/8
        detail = detail1 * torch.sigmoid(context1) # H/8, W/8
        context = detail2 * torch.sigmoid(context2) # H/32, W/32
        context = self.upconv_second(context) # H/32, W/32 -> H/8, W/8
        # context = F.interpolate(
        #     context, scale_factor=4, mode='bilinear', align_corners=False) # H/32, W/32 -> H/8, W/8
        # context = F.interpolate(
        #     context, size=dsize, mode='bilinear', align_corners=True) # H/32, W/32 -> H/8, W/8
        out = self.conv(detail + context)
        return out
    
# Input left (H,W), input right (H/4, W/4) -> output (H, W)
AggregationLayer_full_quarter_full = AggregationLayer

# Input left (H,W), input right (H/2, W/2) -> output (H/2, W/2)
class AggregationLayer_full_half_half(AggregationLayer):
    def __init__(self, channels = 128):
        super().__init__(channels = channels)
        
        self.left1 = nn.Sequential(
            nn.Conv2d(
                self.channels, self.channels, kernel_size=3, stride=1,
                padding=1, groups=self.channels, bias=False), # H, W
            nn.BatchNorm2d(self.channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False) # H/2, W/2
            # nn.Conv2d(
            #     128, 128, kernel_size=1, stride=1,
            #     padding=0, bias=False), # H, W
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                self.channels, self.channels, kernel_size=3, stride=2,
                padding=1, bias=False), # H/2, W/2
            nn.BatchNorm2d(self.channels),
            # nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False) # H/4, W/4
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                64, self.channels, kernel_size=3, stride=1,
                padding=1, bias=False), # H, W
            nn.BatchNorm2d(self.channels),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                64, self.channels, kernel_size=3, stride=1,
                padding=1, groups=64, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.Conv2d(
                self.channels, self.channels, kernel_size=1, stride=1,
                padding=0, bias=False), # H, W
        )
        # ##TODO: does this really has no relu?
        # self.conv = nn.Sequential(
        #     nn.Conv2d(
        #         128, 128, kernel_size=3, stride=1,
        #         padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),  # not shown in paper
        # )

    def forward(self, x_d, x_s): # [W/8, H/8], [W/16, H/16] -> [W/16, H/16]
        # print(f"Aggregation layer input sizes: x_d: {x_d.size()}, x_s: {x_s.size()}")
        # x_d: torch.Size([16, 128, 40, 92]), x_s: torch.Size([16, 128, 10, 23])
        dsize = x_d.size()[2:] # 40, 92 = H/8, W/8
        detail1 = self.left1(x_d) # Conv2d + BN + AvgPool: H/16, W/16
        detail2 = self.left2(x_d) # Conv2d + BN + AvgPool: H/16, W/16
        context1 = self.right1(x_s) # Conv2d + BN: H/16, W/16 # RuntimeError: Given groups=1, weight of size [128, 128, 3, 3], expected input[24, 64, 20, 46] to have 128 channels, but got 64 channels instead
        context2 = self.right2(x_s) # Conv2d + BN + Conv2d: H/16, W/16
        # print(f"Aggregation layer context sizes: context1: {context1.size()}, context2: {context2.size()}")
        # print(f"Aggregation layer detail sizes: detail1: {detail1.size()}, detail2: {detail2.size()}")
        # context1 = F.interpolate(
        #     context1, size=dsize, mode='bilinear', align_corners=True) # H/16, W/16 -> H/8, W/8
        detail = detail1 * torch.sigmoid(context1) # H/16, W/16
        context = detail2 * torch.sigmoid(context2) # H/16, W/16
        # context = F.interpolate(
        #     context, size=dsize, mode='bilinear', align_corners=True) # H/16, W/16 -> H/8, W/8
        out = self.conv(detail + context)
        return out # H/16, W/16
    
# Input left (H,W), input right (H/2, W/2) -> output (H/2, W/2)
class AggregationLayer_full_quarter_half(AggregationLayer):
    def __init__(self, channels = 128):
        super().__init__(channels = channels)
        
        # self.left1 = nn.Sequential(
        #     nn.Conv2d(
        #         self.channels, self.channels, kernel_size=3, stride=1,
        #         padding=1, groups=self.channels, bias=False), # H, W
        #     nn.BatchNorm2d(self.channels),
        #     # nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False) # H/2, W/2
        #     nn.Conv2d(
        #         128, 128, kernel_size=1, stride=1,
        #         padding=0, bias=False), # H, W
        # )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                self.channels, self.channels, kernel_size=3, stride=2,
                padding=1, bias=False), # H/2, W/2
            nn.BatchNorm2d(self.channels),
            # nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False) # H/4, W/4
        )
        # self.right1 = nn.Sequential(
        #     nn.Conv2d(
        #         128, 128, kernel_size=3, stride=1,
        #         padding=1, bias=False), # H, W
        #     nn.BatchNorm2d(128),
        # )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                self.channels, 128, kernel_size=3, stride=1,
                padding=1, groups=self.channels, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, self.channels, kernel_size=1, stride=1,
                padding=0, bias=False), # H, W
        )

        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(
                self.channels, self.channels, kernel_size=2, stride=2,
                padding=0, bias=False), # H/2, W/2 -> H, W
            nn.BatchNorm2d(self.channels),
        )
        # ##TODO: does this really has no relu?
        # self.conv = nn.Sequential(
        #     nn.Conv2d(
        #         128, 128, kernel_size=3, stride=1,
        #         padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),  # not shown in paper
        # )

    def forward(self, x_d, x_s): # [W/8, H/8], [W/32, H/32] -> [W/16, H/16]
        # print(f"Aggregation layer input sizes: x_d: {x_d.size()}, x_s: {x_s.size()}")
        # x_d: torch.Size([16, 128, 40, 92]), x_s: torch.Size([16, 128, 10, 23])
        dsize = x_d.size()[2:] # 40, 92 = H/8, W/8
        dsize = (dsize[0]//2, dsize[1]//2) # H/16, W/16
        # detail1 = self.left1(x_d) # Conv2d + BN + Conv2d: H/8, W/8
        detail2 = self.left2(x_d) # Conv2d + BN + AvgPool: H/16, W/16
        # context1 = self.right1(x_s) # Conv2d + BN: H/32, W/32
        context2 = self.right2(x_s) # Conv2d + BN + Conv2d: H/32, W/32
        # context1 = F.interpolate(
        #     context1, size=dsize, mode='bilinear', align_corners=True) # H/32, W/32 -> H/16, W/16
        # detail = detail1 * torch.sigmoid(context1) # H/16, W/16
        # context = detail2 * torch.sigmoid(context2) # H/16, W/16
        # context = F.interpolate(
        #     context, size=dsize, mode='bilinear', align_corners=True) # H/16, W/16 -> H/8, W/8
        # out = self.conv(detail + context)

        # context2 = F.interpolate(
        #     context2, size=dsize, mode='bilinear', align_corners=True) # H/32, W/32 -> H/16, W/16
        # context2 = F.interpolate(
        #     context2, scale_factor=2, mode='bilinear', align_corners=False) # H/32, W/32 -> H/16, W/16
        # TransposedConv instead of interpolate
        context2 = self.upconv(context2) # H/32, W/32 -> H/16, W/16
        context = detail2 * torch.sigmoid(context2) # H/16, W/16
        
        out = self.conv(context)
        return out # H/16, W/16

class Head(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, conv="2D"):
        super(Head, self).__init__()

        self.conv_type=  conv
        self.in_chan= in_chan
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1, conv=conv)
        # self.conv_m = ConvBNReLU(mid_chan, mid_chan, 3, stride=1)
        # self.drop = nn.Dropout(0.1)
        if conv=="2D":
            self.conv_out = nn.Conv2d(
                mid_chan, n_classes, kernel_size=1, stride=1,
                padding=0, bias=True)
        else:
            self.conv_out = nn.Conv3d(
                mid_chan, n_classes, kernel_size=1, stride=1,
                padding=0, bias=True)
            

    def forward(self, x, size=None):
        feat = self.conv(x)
        # feat = self.drop(feat)
        # feat = self.conv_m(feat)
        feat = self.conv_out(feat)
        # if not size is None:
        #     if self.conv_type=="2D":
        #         feat = F.interpolate(feat, size=size,
        #                              mode='bilinear', align_corners=True)
            # else:
            #     F.interpolate(feat, size=[feat.shape[2], size[0], size[1]], mode='trilinear', align_corners=True)
        return feat
    
class ContextGuidedUpsampleBlock(nn.Module):
    """Context-guided upsample by 2x using learned weights for unfold operation."""
    
    def __init__(self, in_chan, guide_chan, out_chan, scale_factor=2):
        super(ContextGuidedUpsampleBlock, self).__init__()
        self.scale_factor = scale_factor
        
        # Generate upsampling weights from guide
        self.weight_net = nn.Sequential(
            nn.Conv2d(guide_chan, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 9, kernel_size=1, bias=False),  # 3x3 kernel weights
            nn.Softmax(dim=1)  # normalize weights across 9 positions
        )
        
        # Refinement after upsampling
        self.refine = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, guide):
        """
        Args:
            x: low-res feature [B, in_chan, H, W]
            guide: high-res guide [B, guide_chan, H*scale, W*scale]
        Returns:
            upsampled feature [B, out_chan, H*scale, W*scale]
        """
        b, c, h, w = x.shape
        
        # Generate upsampling weights from guide
        up_weights = self.weight_net(guide)  # [B, 9, H*scale, W*scale]
        
        # Unfold low-res features into 3x3 patches
        x_unfold = F.unfold(x, kernel_size=3, padding=1)  # [B, in_chan*9, H*W]
        x_unfold = x_unfold.view(b, c, 9, h, w)  # [B, in_chan, 9, H, W]
        
        # Upsample unfolded features
        x_unfold = x_unfold.permute(0, 1, 3, 4, 2)  # [B, in_chan, H, W, 9]
        x_unfold = x_unfold.reshape(b, c * h * w, 9)  # [B, in_chan*H*W, 9]
        x_unfold = x_unfold.permute(0, 2, 1)  # [B, 9, in_chan*H*W]
        x_unfold = x_unfold.reshape(b, 9, c, h, w)  # [B, 9, in_chan, H, W]
        x_unfold = x_unfold.permute(0, 2, 1, 3, 4)  # [B, in_chan, 9, H, W]
        
        # Nearest neighbor upsample
        # todo: chenge this to scale_factor instead of size
        x_unfold = F.interpolate(
            x_unfold.reshape(b * c, 9, h, w),
            size=(h * self.scale_factor, w * self.scale_factor),
            mode='nearest'
        )  # [B*in_chan, 9, H*scale, W*scale]
        x_unfold = x_unfold.view(b, c, 9, h * self.scale_factor, w * self.scale_factor)
        
        # Apply context weights: weighted sum over 9 positions
        up_weights = up_weights.unsqueeze(1)  # [B, 1, 9, H*scale, W*scale]
        x_up = (x_unfold * up_weights).sum(2)  # [B, in_chan, H*scale, W*scale]
        
        # Refine upsampled features
        out = self.refine(x_up)  # [B, out_chan, H*scale, W*scale]
        
        return out


class GuidedUpsampleBlock(nn.Module):
    """Upsample spatial dimensions by 2x using guide features at target resolution."""
    
    def __init__(self, in_chan, guide_chan, out_chan):
        super(GuidedUpsampleBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_chan, in_chan, kernel_size=4, stride=2, 
            padding=1, bias=False)
        self.guide_conv = nn.Sequential(
            nn.Conv2d(guide_chan, in_chan, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_chan * 2, out_chan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, guide):
        """
        Args:
            x: low-res feature [B, in_chan, H, W]
            guide: high-res guide [B, guide_chan, H*2, W*2]
        Returns:
            upsampled feature [B, out_chan, H*2, W*2]
        """
        x_up = self.upsample(x)  # [B, in_chan, H*2, W*2]
        guide_feat = self.guide_conv(guide)  # [B, in_chan, H*2, W*2]: TypeError: conv2d() received an invalid combination of arguments - got (list, Parameter, NoneType, tuple, tuple, tuple, int), but expected one of:
#  * (Tensor input, Tensor weight, Tensor bias = None, tuple of ints stride = 1, tuple of ints padding = 0, tuple of ints dilation = 1, int groups = 1)
#       didn't match because some of the arguments have invalid types: (list of [Tensor], Parameter, NoneType, tuple of (int, int), tuple of (int, int), tuple of (int, int), int)
        fused = torch.cat([x_up, guide_feat], dim=1)  # [B, in_chan*2, H*2, W*2]
        out = self.fuse(fused)  # [B, out_chan, H*2, W*2]
        return out

class UpsampleBlockInterpolate(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(UpsampleBlock, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1, groups=out_chan, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )
    def forward(self, x, dummy):
        x = self.project(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.refine(x)

class UpsampleBlock(nn.Module):
    """Upsample spatial dimensions by 2x using transposed convolution + refinement."""
    
    def __init__(self, in_chan, out_chan):
        super(UpsampleBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_chan, out_chan, kernel_size=4, stride=2, 
                padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_chan, out_chan, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, dummy):
        return self.up(x)

class FeatureExtractionNet(nn.Module):

    def __init__(self, cfgs = None, pretrained: bool = True, checkpoint_path: str = None):
        super(FeatureExtractionNet, self).__init__()
        
        self.cfgs = cfgs
        # self.output_channels = channels[::-1]
        # self.output_channels = [80, 160, 320] # for LightStereo heads at H/4, H/8, H/16
        self.output_channels = cfgs.get('OUTPUT_CHANNELS', None) #, [-1,-1,-1]) # for LightStereo heads at H/4, H/8, H/16
        # self.output_channels = cfgs.get('INTERMEDIATE_CHANNELS', [80, 160, 320]) # for LightStereo heads at H/4, H/8, H/16
        self.head_intermediate_channels = cfgs.get('HEAD_INTERMEDIATE_CHANNELS', None) # for LightStereo heads at H/4, H/8, H/16
        self.aggregation_channels = cfgs.get('AGGREGATION_CHANNELS', None) # for LightStereo aggregation layers
        
        if self.output_channels is None:
            raise ValueError("FeatureExtractionNet: BACKBONE_CFGS.OUTPUT_CHANNELS must be specified in cfgs")
        if self.head_intermediate_channels is None:
            raise ValueError("FeatureExtractionNet: BACKBONE_CFGS.HEAD_INTERMEDIATE_CHANNELS must be specified in cfgs")
        if self.aggregation_channels is None:
            raise ValueError("FeatureExtractionNet: BACKBONE_CFGS.AGGREGATION_CHANNELS must be specified in cfgs")

        self.detail = ShallowBranch(self.aggregation_channels) # (W/2 W/4, W/8) [W/8, H/8]
        self.segment = DeepBranch(self.aggregation_channels) # [W/4, H/4 | W/8, H/8 | W/16, H/16 | W/32, H/32 | W/32, H/32]
        # self.aggregate_4 = AggregationLayer_full_half_half(channels=self.aggregation_channels)
        # self.aggregate_4 = AggregationLayer_full_half_full(channels=self.aggregation_channels)
        # self.aggregate_4 = AggregationLayer_full_quarter_full(channels=self.aggregation_channels)
        self.aggregate_8 = AggregationLayer_full_quarter_full(channels=self.aggregation_channels)
        self.aggregate_16 = AggregationLayer_full_quarter_half(channels=self.aggregation_channels)

        self.upsample_type = cfgs.get('UPSAMPLE_TYPE', None) # 'guided' or 'context_guided' or 'simple'
        if self.upsample_type is None:
            raise ValueError("FeatureExtractionNet: BACKBONE_CFGS.UPSAMPLE_TYPE must be specified in cfgs")

        if self.upsample_type == 'guided':
            self.upsample = GuidedUpsampleBlock(self.aggregation_channels, self.aggregation_channels, self.aggregation_channels)
        elif self.upsample_type == 'context_guided':
            self.upsample = ContextGuidedUpsampleBlock(self.aggregation_channels, self.aggregation_channels, self.aggregation_channels)
        elif self.upsample_type == 'interpolate':
            self.upsample = UpsampleBlockInterpolate(self.aggregation_channels, self.aggregation_channels)
        elif self.upsample_type == 'simple':
            self.upsample = UpsampleBlock(self.aggregation_channels, self.aggregation_channels)
        else:
            raise ValueError(f"FeatureExtractionNet: BACKBONE_CFGS.UPSAMPLE_TYPE has invalid value: {self.upsample_type}")

        self.head_4 = Head(self.aggregation_channels, self.head_intermediate_channels[0], self.output_channels[0]) # just 2D conv
        self.head_8 = Head(self.aggregation_channels, self.head_intermediate_channels[1], self.output_channels[1]) # just 2D conv
        self.head_16 = Head(self.aggregation_channels, self.head_intermediate_channels[2], self.output_channels[2]) # just 2D conv


        self.init_weights()

    # For LightStereo I need: H/4, H/8, H/16
    def forward(self, x):


        feat_d = self.detail(x) 
        # (available: W/2, W/4, W/8) returns: [W/8, H/8]
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x) 
        #(available: W/4, W/8, W/16, W/32, W/32) returns: [W/4, H/4 | W/8, H/8 | W/16, H/16 | W/32, H/32 | W/32, H/32]

        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"feat_d[0] Found NaN!", force_print=feat_d[0].isfinite().all()==False)
            debug_printer.print_of_function_force_print(lambda : f"feat_d[1] Found NaN!", force_print=feat_d[1].isfinite().all()==False)
            debug_printer.print_of_function_force_print(lambda : f"feat_d[2] Found NaN!", force_print=feat_d[2].isfinite().all()==False)
            debug_printer.print_of_function_force_print(lambda : f"feat_2 Found NaN!", force_print=feat2.isfinite().all()==False)
            debug_printer.print_of_function_force_print(lambda : f"feat_3 Found NaN!", force_print=feat3.isfinite().all()==False)
            debug_printer.print_of_function_force_print(lambda : f"feat_4 Found NaN!", force_print=feat4.isfinite().all()==False)
            debug_printer.print_of_function_force_print(lambda : f"feat_5_4 Found NaN!", force_print=feat5_4.isfinite().all()==False)
            debug_printer.print_of_function_force_print(lambda : f"feat_s Found NaN!", force_print=feat_s.isfinite().all()==False)


        # print(f"feat_d sizes: {[fd.size() for fd in feat_d]}")
        # print(f"feat_s size: {feat_s.size()}")
        # print(f"feat5_4 size: {feat5_4.size()}")
        # print(f"feat4 size: {feat4.size()}")
        # feat_d sizes: [torch.Size([24, 128, 40, 92])] 1/8
        # feat_s size: torch.Size([24, 128, 10, 23]) 1/32
        # feat5_4 size: torch.Size([24, 128, 10, 23]) 1/32
        # feat4 size: torch.Size([24, 64, 20, 46]) 1/16
        
        # H/4
        # feat_head_4 = self.aggregate_4(feat_d[-2], feat5_4) # [W/4, H/4], [W/16, H/16] -> [W/4, H/4]
        # feat_head_4 = self.aggregate_4(feat_d[-1], feat4) # [W/8, H/8], [W/16, H/16] -> [W/8, H/8]
        # feat_head_4 = self.upsample(feat_head_4, feat_d[-1]) # [W/4, H/4] # main feature for cost volume

        # H/8 # original
        feat_head_8 = self.aggregate_8(feat_d[-1], feat_s) # [W/8, H/8], [W/32, H/32] -> [W/8, H/8]


        # print(f"feat_head_8 size before upsample: {feat_head_8.size()}") # 1/16
        feat_head_4 = self.upsample(feat_head_8, feat_d[-2]) # [H/8, W/8], [W/4, H/4] -> [W/4, H/4]
        # print(f"feat_head_4 size after upsample: {feat_head_4.size()}")
        # H/16
        feat_head_16 = self.aggregate_16(feat_d[-1], feat5_4) # [W/8, H/8], [W/32, H/32] -> [W/16, H/16]

        aggregated_feats_4 = self.head_4(feat_head_4) # [B, 80, H/4, W/4]
        aggregated_feats_8 = self.head_8(feat_head_8) # [B, 160, H/8, W/8]
        aggregated_feats_16 = self.head_16(feat_head_16) # [B, 320, H/16, W/16]

        # print(f"aggregated_feats_4 size: {aggregated_feats_4.size()}")
        # print(f"aggregated_feats_8 size: {aggregated_feats_8.size()}")
        # print(f"aggregated_feats_16 size: {aggregated_feats_16.size()}")

        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"aggregated_feats_4 Found NaN!", force_print=aggregated_feats_4.isfinite().all()==False)
            debug_printer.print_of_function_force_print(lambda : f"aggregated_feats_8 Found NaN!", force_print=aggregated_feats_8.isfinite().all()==False)
            debug_printer.print_of_function_force_print(lambda : f"aggregated_feats_16 Found NaN!", force_print=aggregated_feats_16.isfinite().all()==False)

        # return [aggregated_feats_4.half(), aggregated_feats_8.half(), aggregated_feats_16.half()] # W/4, W/8, W/16
        return [aggregated_feats_4, aggregated_feats_8, aggregated_feats_16] # W/4, W/8, W/16
        # actually I get: 1/8, 1/8, 1/16

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

class FeatureExtractionNetDeepOnly(nn.Module):

    def __init__(self, cfgs = None, pretrained: bool = True, checkpoint_path: str = None):
        super(FeatureExtractionNet, self).__init__()
        
        self.cfgs = cfgs
        # self.output_channels = channels[::-1]
        # self.output_channels = [80, 160, 320] # for LightStereo heads at H/4, H/8, H/16
        self.output_channels = cfgs.get('OUTPUT_CHANNELS', None) # for LightStereo heads at H/4, H/8, H/16
        # self.output_channels = cfgs.get('INTERMEDIATE_CHANNELS', [80, 160, 320]) # for LightStereo heads at H/4, H/8, H/16
        self.head_intermediate_channels = cfgs.get('HEAD_INTERMEDIATE_CHANNELS', None) # for LightStereo heads at H/4, H/8, H/16
        self.aggregation_channels = cfgs.get('AGGREGATION_CHANNELS', None) # for LightStereo aggregation layers
        
        if self.output_channels is None:
            raise ValueError("FeatureExtractionNet: BACKBONE_CFGS.OUTPUT_CHANNELS must be specified in cfgs")
        if self.head_intermediate_channels is None:
            raise ValueError("FeatureExtractionNet: BACKBONE_CFGS.HEAD_INTERMEDIATE_CHANNELS must be specified in cfgs")
        if self.aggregation_channels is None:
            raise ValueError("FeatureExtractionNet: BACKBONE_CFGS.AGGREGATION_CHANNELS must be specified in cfgs")

        # self.detail = ShallowBranch(self.aggregation_channels) # (W/2 W/4, W/8) [W/8, H/8]
        self.segment = DeepBranch(self.aggregation_channels) # [W/4, H/4 | W/8, H/8 | W/16, H/16 | W/32, H/32 | W/32, H/32]
        # self.aggregate_4 = AggregationLayer_full_half_half(channels=self.aggregation_channels)
        # self.aggregate_4 = AggregationLayer_full_half_full(channels=self.aggregation_channels)
        # self.aggregate_4 = AggregationLayer_full_quarter_full(channels=self.aggregation_channels)
        # self.aggregate_8 = AggregationLayer_full_quarter_full(channels=self.aggregation_channels)
        # self.aggregate_16 = AggregationLayer_full_quarter_half(channels=self.aggregation_channels)

        # self.upsample_type = cfgs.get('UPSAMPLE_TYPE', 'simple') # 'guided' or 'context_guided' or 'simple'
        # if self.upsample_type == 'guided':
        #     self.upsample = GuidedUpsampleBlock(self.aggregation_channels, self.aggregation_channels, self.aggregation_channels)
        # elif self.upsample_type == 'context_guided':
        #     self.upsample = ContextGuidedUpsampleBlock(self.aggregation_channels, self.aggregation_channels, self.aggregation_channels)
        # elif self.upsample_type == 'interpolate':
        #     self.upsample = UpsampleBlockInterpolate(self.aggregation_channels, self.aggregation_channels)
        # else:
        #     self.upsample = UpsampleBlock(self.aggregation_channels, self.aggregation_channels)

        self.head_4 = Head(self.aggregation_channels, self.head_intermediate_channels[0], self.output_channels[0]) # just 2D conv
        self.head_8 = Head(self.aggregation_channels, self.head_intermediate_channels[1], self.output_channels[1]) # just 2D conv
        self.head_16 = Head(self.aggregation_channels, self.head_intermediate_channels[2], self.output_channels[2]) # just 2D conv


        self.init_weights()

    # For LightStereo I need: H/4, H/8, H/16
    # todo: maybe instead should make segment net as a encoder-decpder
    def forward(self, x):


        # feat_d = self.detail(x) 
        # (available: W/2, W/4, W/8) returns: [W/8, H/8]
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x) 
        #(available: W/4, W/8, W/16, W/32, W/32) returns: [W/4, H/4 | W/8, H/8 | W/16, H/16 | W/32, H/32 | W/32, H/32]

        if self.training:
            # debug_printer.print_of_function_force_print(lambda : f"feat_d[0] Found NaN!", force_print=feat_d[0].isfinite().all()==False)
            # debug_printer.print_of_function_force_print(lambda : f"feat_d[1] Found NaN!", force_print=feat_d[1].isfinite().all()==False)
            # debug_printer.print_of_function_force_print(lambda : f"feat_d[2] Found NaN!", force_print=feat_d[2].isfinite().all()==False)
            debug_printer.print_of_function_force_print(lambda : f"feat_2 Found NaN!", force_print=feat2.isfinite().all()==False)
            debug_printer.print_of_function_force_print(lambda : f"feat_3 Found NaN!", force_print=feat3.isfinite().all()==False)
            debug_printer.print_of_function_force_print(lambda : f"feat_4 Found NaN!", force_print=feat4.isfinite().all()==False)
            debug_printer.print_of_function_force_print(lambda : f"feat_5_4 Found NaN!", force_print=feat5_4.isfinite().all()==False)
            debug_printer.print_of_function_force_print(lambda : f"feat_s Found NaN!", force_print=feat_s.isfinite().all()==False)


        # print(f"feat_d sizes: {[fd.size() for fd in feat_d]}")
        # print(f"feat_s size: {feat_s.size()}")
        # print(f"feat5_4 size: {feat5_4.size()}")
        # print(f"feat4 size: {feat4.size()}")
        # feat_d sizes: [torch.Size([24, 128, 40, 92])] 1/8
        # feat_s size: torch.Size([24, 128, 10, 23]) 1/32
        # feat5_4 size: torch.Size([24, 128, 10, 23]) 1/32
        # feat4 size: torch.Size([24, 64, 20, 46]) 1/16
        
        # H/4
        # feat_head_4 = self.aggregate_4(feat_d[-2], feat5_4) # [W/4, H/4], [W/16, H/16] -> [W/4, H/4]
        # feat_head_4 = self.aggregate_4(feat_d[-1], feat4) # [W/8, H/8], [W/16, H/16] -> [W/8, H/8]
        # feat_head_4 = self.upsample(feat_head_4, feat_d[-1]) # [W/4, H/4] # main feature for cost volume

        # H/8 # original
        # feat_head_8 = self.aggregate_8(feat_d[-1], feat_s) # [W/8, H/8], [W/32, H/32] -> [W/8, H/8]


        # # print(f"feat_head_8 size before upsample: {feat_head_8.size()}") # 1/16
        # feat_head_4 = self.upsample(feat_head_8, feat_d[-2]) # [H/8, W/8], [W/4, H/4] -> [W/4, H/4]
        # # print(f"feat_head_4 size after upsample: {feat_head_4.size()}")
        # # H/16
        # feat_head_16 = self.aggregate_16(feat_d[-1], feat5_4) # [W/8, H/8], [W/32, H/32] -> [W/16, H/16]

        aggregated_feats_4 = self.head_4(feat2) # [B, 80, H/4, W/4]
        aggregated_feats_8 = self.head_8(feat3) # [B, 160, H/8, W/8]
        aggregated_feats_16 = self.head_16(feat4) # [B, 320, H/16, W/16]

        # print(f"aggregated_feats_4 size: {aggregated_feats_4.size()}")
        # print(f"aggregated_feats_8 size: {aggregated_feats_8.size()}")
        # print(f"aggregated_feats_16 size: {aggregated_feats_16.size()}")

        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"aggregated_feats_4 Found NaN!", force_print=aggregated_feats_4.isfinite().all()==False)
            debug_printer.print_of_function_force_print(lambda : f"aggregated_feats_8 Found NaN!", force_print=aggregated_feats_8.isfinite().all()==False)
            debug_printer.print_of_function_force_print(lambda : f"aggregated_feats_16 Found NaN!", force_print=aggregated_feats_16.isfinite().all()==False)

        return [aggregated_feats_4, aggregated_feats_8, aggregated_feats_16] # W/4, W/8, W/16
        # actually I get: 1/8, 1/8, 1/16

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


def leanbackbone():
    pass

def count_model_params(model):
    num_params = sum(p.numel() for p in model.parameters())
    return num_params