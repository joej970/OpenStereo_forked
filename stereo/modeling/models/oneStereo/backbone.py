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
        low = self.deconv(low)
        feat = torch.cat([high, low], 1)
        feat = self.conv(feat)
        return feat


class Backbone(nn.Module):
    def __init__(self, backbone='MobileNetv2', pretrained=None, checkpoint_path=None):
        super().__init__()
        self.backbone = backbone
        self.pretrained = pretrained if pretrained is not None else True  # Whether to use pretrained weights
        self.checkpoint_path = checkpoint_path if checkpoint_path is not None else None  # Path to the checkpoint file, if provided

        message = f"Using backbone: {self.backbone}, pretrained: {self.pretrained}, checkpoint_path: {self.checkpoint_path}"
        print(message)

        if backbone == 'MobileNetv2':
            # model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True)
            model = timm.create_model('mobilenetv2_100', pretrained=self.pretrained)
            channels = [160, 96, 32, 24]
        elif backbone == 'EfficientNetv2':
            # model = timm.create_model('efficientnetv2_rw_s', pretrained=True, features_only=True)
            model = timm.create_model('efficientnetv2_rw_s', pretrained=self.pretrained)
            channels = [272, 160, 64, 48]
        
        elif backbone == 'mobileone_s0_local':  # Use local MobileOne implementation
            model = mobileone(num_classes=1000, inference_mode=False, variant="s0")
            
            # Load pretrained weights from .pth file
            # Option 1: Load unfused checkpoint (recommended for training)
            if self.checkpoint_path is not None:
                if self.pretrained is False:
                    print("")
                    print("WARNING: You supplied pretrained model but set pretrained to false.")
                    print("")
                    
                # If you have a specific checkpoint path, use it
                # Otherwise, you can use the default path below
                
                # self.checkpoint_path = '/d/hpc/home/zr1677/OpenStereo_forked/stereo/modeling/models/oneStereo/mobileone_s0_unfused.pth'
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                # model.load_state_dict(checkpoint, strict=False)  # strict=False in case of minor key mismatches
                model.load_state_dict(checkpoint, strict=True)  # strict=False in case of minor key mismatches
            else:
                print("No checkpoint path provided, training from zero.")
            # Option 2: If you want to use fused checkpoint for inference
            # checkpoint_path = "path/to/mobileone_s0_fused.pth"
            # checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # model = reparameterize_model(model)  # Convert to inference mode first
            # model.load_state_dict(checkpoint, strict=False)
            
            channels = [1024, 256, 128, 48] 

            # For local MobileOne, we need to access the stages directly
            # The model structure: stage0 -> stage1 -> stage2 -> stage3 -> stage4
            self.stage0 = model.stage0  # Initial conv block
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

            return

        elif backbone == 'mobileone_s0': # needs timm 0.9.* (default is 0.4.12); print(timm.list_models('*one*'))
            model = timm.create_model('mobileone_s0', pretrained=True, features_only=True)
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

            # for name, param in model.named_parameters():
            #     print(f"{name}: requires_grad={param.requires_grad}")

            return

            # [{'num_chs': 48, 'reduction': 2, 'module': 'stem'}, {'num_chs': 48, 'reduction': 4, 'module': 'stages.0'}, {'num_chs': 128, 'reduction': 8, 'module': 'stages.1'}, {'num_chs': 256, 'reduction': 16, 'module': 'stages.2'}, {'num_chs': 1024, 'reduction': 32, 'module': 'final_conv'}]
        else:
            raise NotImplementedError

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        # self.act1 = model.act1 # in timm 0.9.* act is included in bn1
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

        if self.backbone == 'mobileone_s0':
            c1 = self.conv_stem(images)  # [bz, 48, H/2, W/2]
            c1 = self.stage0(c1) # [bz, 48, H/4, W/4]
            c2 = self.stage1(c1) # [bz, 128, H/8, W/8]
            c3 = self.stage2(c2) # [bz, 256, H/16, W/16]
            c4 = self.stage3(c3) # [bz, 1024, H/32, W/32]

            p4 = self.fpn_layer4(c4, c3)  # 1024->256: [bz, 256, H/16, W/16]
            p3 = self.fpn_layer3(p4, c2)  # 256->128 [bz, 128, H/8, W/8]
            p2 = self.fpn_layer2(p3, c1)  # 128->48 [bz, 48, H/4, W/4]
            p2 = self.out_conv(p2)        # 48->48 [bz, 48, H/4, W/4]

            return [p2, p3, p4, c4] # [bz, 48, H/4, W/4], [bz, 128, H/8, W/8], [bz, 256, H/16, W/16], [bz, 1024, H/32, W/32]

        elif self.backbone == 'mobileone_s0_local':
            c1 = self.stage0(images)  # [bz, 48, H/2, W/2] - stage0 is the initial conv
            c1 = self.stage1(c1) # [bz, 48, H/4, W/4]
            c2 = self.stage2(c1) # [bz, 128, H/8, W/8]
            c3 = self.stage3(c2) # [bz, 256, H/16, W/16]
            c4 = self.stage4(c3) # [bz, 1024, H/32, W/32]

            p4 = self.fpn_layer4(c4, c3)  # 1024->256: [bz, 256, H/16, W/16]
            p3 = self.fpn_layer3(p4, c2)  # 256->128 [bz, 128, H/8, W/8]
            p2 = self.fpn_layer2(p3, c1)  # 128->48 [bz, 48, H/4, W/4]
            p2 = self.out_conv(p2)        # 48->48 [bz, 48, H/4, W/4]

            return [p2, p3, p4, c4] # [bz, 48, H/4, W/4], [bz, 128, H/8, W/8], [bz, 256, H/16, W/16], [bz, 1024, H/32, W/32]

        else:
            # c1 = self.act1(self.bn1(self.conv_stem(images)))  # [bz, 32, H/2, W/2]
            c1 = self.bn1(self.conv_stem(images))  # [bz, 32, H/2, W/2]  # in timm 0.9.* act is included in bn1
            c1 = self.block0(c1)  # [bz, 16, H/2, W/2]
            c2 = self.block1(c1)  # [bz, 24, H/4, W/4]
            c3 = self.block2(c2)  # [bz, 32, H/8, W/8]
            c4 = self.block3(c3)  # [bz, 96, H/16, W/16]
            c5 = self.block4(c4)  # [bz, 160, H/32, W/32]

            p4 = self.fpn_layer4(c5, c4)  # [bz, 96, H/16, W/16]
            p3 = self.fpn_layer3(p4, c3)  # [bz, 32, H/8, W/8]
            p2 = self.fpn_layer2(p3, c2)  # [bz, 24, H/4, W/4]
            p2 = self.out_conv(p2)

            return [p2, p3, p4, c5]
