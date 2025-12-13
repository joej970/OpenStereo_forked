import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self):
        super(ShallowBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2), # W/2, H/2
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2), # W/4, H/4
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2), # W/8, H/8
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

    def forward(self, x):
        feat = self.S1(x) # W/2, H/2
        feat = self.S2(feat) # W/4, H/4
        feat = self.S3(feat) # W/8, H/8
        return feat # W/8, H/8

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

    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        # TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

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
        return feat

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
        )
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
        return feat

class DeepBranch(nn.Module):

    def __init__(self):
        super(DeepBranch, self).__init__()
        self.S1S2 = StemBlock() # S1= 1/2 S2= 1/4
        self.S3 = nn.Sequential( 
            GELayerS2(16, 32),
            GELayerS1(32, 32),
        )
        self.S4 = nn.Sequential(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock() # no change in size

    def forward(self, x):
        feat2 = self.S1S2(x)  # S1= 1/2 S2= 1/4
        feat3 = self.S3(feat2)  # S3= 1/8
        feat4 = self.S4(feat3)  # S4= 1/16
        feat5_4 = self.S5_4(feat4)  # S5_4= 1/32
        feat5_5 = self.S5_5(feat5_4) # 1/32
        return feat2, feat3, feat4, feat5_4, feat5_5 # [W/4, H/4 | W/8, H/8 | W/16, H/16 | W/32, H/32 | W/32, H/32]

class AggregationLayer(nn.Module):

    def __init__(self):
        super(AggregationLayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False), # H, W
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False), # H, W
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2,
                padding=1, bias=False), # H/2, W/2
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False) # H/4, W/4
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False), # H, W
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False), # H, W
        )
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # not shown in paper
        )

    def forward(self, x_d, x_s): # [W/8, H/8], [W/32, H/32] -> [W/8, H/8]
        # print(f"Aggregation layer input sizes: x_d: {x_d.size()}, x_s: {x_s.size()}")
        # x_d: torch.Size([16, 128, 40, 92]), x_s: torch.Size([16, 128, 10, 23])
        dsize = x_d.size()[2:] # 40, 92 = H/8, W/8
        left1 = self.left1(x_d) # Con2d + BN + Conv2d: H, W
        left2 = self.left2(x_d) # Conv2d + BN + AvgPool: H/4, W/4
        right1 = self.right1(x_s) # Conv2d + BN: H, W
        right2 = self.right2(x_s) # Conv2d + BN + Conv2d: H, W
        right1 = F.interpolate(
            right1, size=dsize, mode='bilinear', align_corners=True)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = F.interpolate(
            right, size=dsize, mode='bilinear', align_corners=True)
        out = self.conv(left + right)
        return out

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
        if not size is None:
            if self.conv_type=="2D":
                feat = F.interpolate(feat, size=size,
                                     mode='bilinear', align_corners=True)
            # else:
            #     F.interpolate(feat, size=[feat.shape[2], size[0], size[1]], mode='trilinear', align_corners=True)
        return feat

class FeatureExtractionNet(nn.Module):

    def __init__(self):
        super(FeatureExtractionNet, self).__init__()
        self.detail = ShallowBranch() # [W/8, H/8]
        self.segment = DeepBranch() # [W/4, H/4 | W/8, H/8 | W/16, H/16 | W/32, H/32 | W/32, H/32]
        self.aggregate = AggregationLayer()
        self.head = Head(128, 1024, 320) # just 2D conv

        self.init_weights()

    def forward(self, x):
        feat_d = self.detail(x) # 
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x) 
        feat_head = self.aggregate(feat_d, feat_s) # [W/8, H/8], [W/32, H/32] -> [W/8, H/8]

        aggregated_feats = self.head(feat_head) # [B, 320, H/8, W/8]

        return aggregated_feats # [B, 320, H/8, W/8]

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

