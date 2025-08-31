# @Time    : 2024/3/10 10:21
# @Author  : zhangchenming
import timm
import torch
import torch.nn as nn

from functools import partial
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from .mobileone import mobileone, reparameterize_model


class FPNLayer(nn.Module):
    def __init__(self, chan_low, chan_high):
        super().__init__()
        self.deconv = BasicDeconv2d(chan_low, chan_high, kernel_size=4, stride=2, padding=1,
                                    norm_layer=nn.BatchNorm2d,
                                    act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

        self.conv = BasicConv2d(chan_high * 2, chan_high, kernel_size=3, padding=1,
                                norm_layer=nn.BatchNorm2d,
                                act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

    def forward(self, low, high):
        # print(f"FPNLayer: low shape: {low.shape}, high shape: {high.shape}") # low: 160, high: 160
        low = self.deconv(low)
        # print(f"FPNLayer: after deconv low shape: {low.shape}") # 320
        # print(f"FPNLayer: high shape: {high.shape}") # 160
        feat = torch.cat([high, low], dim=1)
        feat = self.conv(feat)
        return feat



class MobileNetV2Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, checkpoint_path: str = None):
        super().__init__()
        print("Creating MobileNetV2Backbone...")
        # model = timm.create_model('mobilenetv2_100', pretrained=pretrained)
        # model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)

        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=False)

        print(f"The model 'mobilenetv2_100' has {count_model_params(model)} parameters")        

        channels = [160, 96, 32, 24]
        print(f"MobileNetV2 Backbone: channels: {channels}")
        # onyl for debugging

        channels = [160, 96, 32, 24]

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1  # in timm 0.9.* act is included in bn1
        self.block0 = model.blocks[0]
        self.block1 = model.blocks[1]
        self.block2 = model.blocks[2]
        self.block3 = model.blocks[3:5]
        self.block4 = model.blocks[5]

        self.fpn_layer4 = FPNLayer(channels[0], channels[1])
        self.fpn_layer3 = FPNLayer(channels[1], channels[2])
        self.fpn_layer2 = FPNLayer(channels[2], channels[3])

        self.out_conv = BasicConv2d(channels[3], channels[3],
                                    kernel_size=3, padding=1, padding_mode="replicate",
                                    norm_layer=nn.InstanceNorm2d)
        self.output_channels = channels[::-1]

    def forward(self, images):
        c1 = self.bn1(self.conv_stem(images))  # [bz, 32, H/2, W/2]
        c1 = self.block0(c1)  # [bz, 16, H/2, W/2]
        c2 = self.block1(c1)  # [bz, 24, H/4, W/4]
        c3 = self.block2(c2)  # [bz, 32, H/8, W/8]
        c4 = self.block3(c3)  # [bz, 96, H/16, W/16]
        c5 = self.block4(c4)  # [bz, 160, H/32, W/32]

        p4 = self.fpn_layer4(c5, c4)  # [bz, 96, H/16, W/16]
        p3 = self.fpn_layer3(p4, c3)  # [bz, 32, H/8, W/8]
        p2 = self.fpn_layer2(p3, c2)  # [bz, 24, H/4, W/4]
        p2 = self.out_conv(p2) # [bz, 24, H/4, W/4]
        return [p2, p3, p4, c5] # [H/4, W/4 | H/8, W/8 | H/16, W/16 | H/32, W/32]

class MobileNetV3SmallBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, checkpoint_path: str = None):
        super().__init__()
        print("Creating MobileNetV3SmallBackbone...")
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=pretrained)

        print("")
        print("")
        print(f"The model 'mobilenetv3_small_100': {self.model}")
        print("")
        print("")

        print(f"The model 'mobilenetv3_small_100' has {count_model_params(self.model)} parameters")

        # channels = [96, 40, 24] # mobilenetv3_small_100
        channels = [96, 40, 24, 16] # mobilenetv3_small_100
        
        self.conv_stem = self.model.conv_stem
        self.bn1 = self.model.bn1  # in timm 0.9.* act is included in bn1 # [bz, 16, H/2, W/2]
        self.block0 = self.model.blocks[0]   # [bz, 16, H/2, W/2] 
        self.block1 = self.model.blocks[1]   # [bz, 24, H/4, W/4]
        self.block2 = self.model.blocks[2]   # [bz, 40, H/8, W/8]
        self.block3 = self.model.blocks[3:5] # [bz, 96, H/16, W/16]

        # FPN layers build top-down
        self.fpn_layer3 = FPNLayer(channels[0], channels[1]) # 96, 40, H/8, W/8
        self.fpn_layer2 = FPNLayer(channels[1], channels[2]) # 40, 24, H/4, W/4
        self.fpn_layer1 = FPNLayer(channels[2], channels[3])
        
        self.out_conv = BasicConv2d(channels[3], channels[3],
                                    kernel_size=3, padding=1, padding_mode="replicate",
                                    norm_layer=nn.InstanceNorm2d) # 24, H/4, W/4
        self.output_channels = channels[::-1]  # [16, 24, 40, 96]

    def forward(self, images):
        c1 = self.bn1(self.conv_stem(images))  # [bz, 16, H/2, W/2]
        c1 = self.block0(c1) # [bz, 16, H/4, W/4]
        c2 = self.block1(c1) # [bz, 24, H/8, W/8]
        c3 = self.block2(c2) # [bz, 40, H/16, W/16]
        c4 = self.block3(c3) # [bz, 96, H/32, W/32]

        p3 = self.fpn_layer3(c4, c3)  # 40, H/16, W/16
        p2 = self.fpn_layer2(p3, c2)  # 24, H/8, W/8
        p1 = self.fpn_layer1(p2, c1)  # 24, H/4, W/4
        p1 = self.out_conv(p1) # 24, H/4, W/4
        return [p1, p2, p3, c4] # H/4, W/4 | H/8, W/8 | H/16, W/16 | H/32, W/32


class MobileNetV3LargeBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, checkpoint_path: str = None):
        super().__init__()
        print("Creating MobileNetV3LargeBackbone...")
        self.model = timm.create_model('mobilenetv3_large_100', pretrained=pretrained)

        # print("")
        # print("")
        # print(f"The model 'mobilenetv3_large_100': {self.model}")
        # print("")
        # print("")

        print(f"The model 'mobilenetv3_large_100' has {count_model_params(self.model)} parameters")

        channels = [160, 112, 40, 24] # mobilenetv3_large_100
        
        self.conv_stem = self.model.conv_stem
        self.bn1 = self.model.bn1  # in timm 0.9.* act is included in bn1 # [bz, 16, H/2, W/2]
        self.block0 = self.model.blocks[0]   # [bz, 16, H/2, W/2]
        self.block1 = self.model.blocks[1]   # [bz, 24, H/4, W/4]
        self.block2 = self.model.blocks[2]   # [bz, 40, H/8, W/8]
        self.block3 = self.model.blocks[3:5] # [bz, 112, H/16, W/16]
        self.block4 = self.model.blocks[5]   # [bz, 160, H/32, W/32]

        # FPN layers build top-down
        self.fpn_layer4 = FPNLayer(channels[0], channels[1]) # 160, 112, H/16, W/16
        self.fpn_layer3 = FPNLayer(channels[1], channels[2]) # 112, 40, H/8, W/8
        self.fpn_layer2 = FPNLayer(channels[2], channels[3]) # 40, 24, H/4, W/4

        self.out_conv = BasicConv2d(channels[3], channels[3],
                                    kernel_size=3, padding=1, padding_mode="replicate",
                                    norm_layer=nn.InstanceNorm2d) # 24, H/4, W/4
        self.output_channels = channels[::-1] # [24, 40, 112, 160]

    def forward(self, images):
        c1 = self.bn1(self.conv_stem(images))  # [bz, 16, H/2, W/2]
        c1 = self.block0(c1) # [bz, 16, H/2, W/2]
        c2 = self.block1(c1) # [bz, 24, H/4, W/4]
        c3 = self.block2(c2) # [bz, 40, H/8, W/8]
        c4 = self.block3(c3) # [bz, 112, H/16, W/16]
        c5 = self.block4(c4) # [bz, 160, H/32, W/32]

        p4 = self.fpn_layer4(c5, c4)  # 112, H/16, W/16
        p3 = self.fpn_layer3(p4, c3)  # 40, H/8, W/8
        p2 = self.fpn_layer2(p3, c2) # 24, H/4, W/4
        p2 = self.out_conv(p2) # 24, H/4, W/4
        return [p2, p3, p4, c5] # H/4, W/4 | H/8, W/8 | H/16, W/16 | H/32, W/32
            
            



class EfficientNetV2Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, checkpoint_path: str = None):
        super().__init__()
        print("Creating EfficientNetV2Backbone...")
        model = timm.create_model('efficientnetv2_rw_s', pretrained=pretrained)
        print(f"The model 'efficientnetv2_rw_s' has {count_model_params(model)} parameters")
        channels = [272, 160, 64, 48]

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1  # in timm 0.9.* act is included in bn1
        self.block0 = model.blocks[0]
        self.block1 = model.blocks[1]
        self.block2 = model.blocks[2]
        self.block3 = model.blocks[3:5]
        self.block4 = model.blocks[5]

        self.fpn_layer4 = FPNLayer(channels[0], channels[1])
        self.fpn_layer3 = FPNLayer(channels[1], channels[2])
        self.fpn_layer2 = FPNLayer(channels[2], channels[3])

        self.out_conv = BasicConv2d(channels[3], channels[3],
                                    kernel_size=3, padding=1, padding_mode="replicate",
                                    norm_layer=nn.InstanceNorm2d)
        self.output_channels = channels[::-1]

    def forward(self, images):
        c1 = self.bn1(self.conv_stem(images))
        c1 = self.block0(c1)
        c2 = self.block1(c1)
        c3 = self.block2(c2)
        c4 = self.block3(c3)
        c5 = self.block4(c4)

        p4 = self.fpn_layer4(c5, c4)
        p3 = self.fpn_layer3(p4, c3)
        p2 = self.fpn_layer2(p3, c2)
        p2 = self.out_conv(p2)
        return [p2, p3, p4, c5]


class MobileOneS0LocalBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, checkpoint_path: str = None):
        super().__init__()
        print("Creating MobileOneS0LocalBackbone...")
        model = mobileone(num_classes=1000, inference_mode=False, variant="s0")
        print(f"The model 'mobileone_s0' (local) has {count_model_params(model)} parameters")
        if checkpoint_path is not None:
            print("Loading checkpoint from:", checkpoint_path)
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint, strict=True)
            print(f"Model loaded on device: {device}.")

        else:
            if pretrained:
                print("WARNING: pretrained=True but no checkpoint_path provided; training from scratch.")
            else:
                print("No checkpoint path provided, training from zero.")

        channels = [1024, 256, 128, 48]
        # stages
        self.stage0 = model.stage0
        self.stage1 = model.stage1
        self.stage2 = model.stage2
        self.stage3 = model.stage3
        self.stage4 = model.stage4

        self.fpn_layer4 = FPNLayer(channels[0], channels[1])
        self.fpn_layer3 = FPNLayer(channels[1], channels[2])
        self.fpn_layer2 = FPNLayer(channels[2], channels[3])

        self.out_conv = BasicConv2d(channels[3], channels[3],
                                    kernel_size=3, padding=1, padding_mode="replicate",
                                    norm_layer=nn.InstanceNorm2d)
        self.output_channels = channels[::-1]

    def forward(self, images):
        c1 = self.stage0(images)
        c1 = self.stage1(c1)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        p4 = self.fpn_layer4(c4, c3)
        p3 = self.fpn_layer3(p4, c2)
        p2 = self.fpn_layer2(p3, c1)
        p2 = self.out_conv(p2)
        return [p2, p3, p4, c4]


class MobileOneS0Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, checkpoint_path: str = None):
        super().__init__()
        print("Creating MobileOneS0Backbone...")
        model = timm.create_model('mobileone_s0', pretrained=pretrained, features_only=True)
        print(f"The model 'mobileone_s0' has {count_model_params(model)} parameters")
        channels = [1024, 256, 128, 48]

        self.conv_stem = model.stem
        self.stage0 = model.stages_0
        self.stage1 = model.stages_1
        self.stage2 = model.stages_2
        self.stage3 = model.stages_3

        self.fpn_layer4 = FPNLayer(channels[0], channels[1])
        self.fpn_layer3 = FPNLayer(channels[1], channels[2])
        self.fpn_layer2 = FPNLayer(channels[2], channels[3])

        self.out_conv = BasicConv2d(channels[3], channels[3],
                                    kernel_size=3, padding=1, padding_mode="replicate",
                                    norm_layer=nn.InstanceNorm2d)
        self.output_channels = channels[::-1]

    def forward(self, images):
        c1 = self.conv_stem(images)
        c1 = self.stage0(c1)
        c2 = self.stage1(c1)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)

        p4 = self.fpn_layer4(c4, c3)
        p3 = self.fpn_layer3(p4, c2)
        p2 = self.fpn_layer2(p3, c1)
        p2 = self.out_conv(p2)
        return [p2, p3, p4, c4]




_BACKBONE_REGISTRY = {
    'MobileNetv2': MobileNetV2Backbone,
    'EfficientNetv2': EfficientNetV2Backbone,
    'mobileone_s0_local': MobileOneS0LocalBackbone,
    'mobileone_s0': MobileOneS0Backbone,
    'mobilenetv3_small_100': MobileNetV3SmallBackbone,
    'mobilenetv3_large_100': MobileNetV3LargeBackbone,
}


def build_backbone(backbone: str = 'MobileNetv2', pretrained: bool = True, checkpoint_path: str = None) -> nn.Module:
    """Single dispatch point to instantiate a backbone class by name."""
    if backbone not in _BACKBONE_REGISTRY:
        raise NotImplementedError(f"Backbone '{backbone}' is not implemented. Available: {list(_BACKBONE_REGISTRY.keys())}")
    return _BACKBONE_REGISTRY[backbone](pretrained=pretrained, checkpoint_path=checkpoint_path)


class Backbone(nn.Module):
    """Thin wrapper kept for backward compatibility. Delegates to a specific backbone implementation."""
    def __init__(self, backbone: str = 'MobileNetv2', pretrained: bool = True, checkpoint_path: str = None):
        super().__init__()
        print(f"Using backbone: {backbone}, pretrained: {pretrained}, checkpoint_path: {checkpoint_path}")
        self.impl = build_backbone(backbone, pretrained=pretrained, checkpoint_path=checkpoint_path)
        self.output_channels = getattr(self.impl, 'output_channels', None)

    def forward(self, images):
        return self.impl(images)


def count_model_params(model):
    num_params = sum(p.numel() for p in model.parameters())
    return num_params

