# @Time    : 2024/3/10 10:21
# @Author  : zhangchenming
import timm
import torch
import torch.nn as nn

from functools import partial
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from .mobileone import mobileone, reparameterize_model

from stereo.modeling.models.oneStereo.leanbackbone import FeatureExtractionNet
from tools.measure import pretty_print_module_counter

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
    def __init__(self, cfgs = None, pretrained: bool = True, checkpoint_path: str = None):
        super().__init__()
        print("Creating MobileNetV2Backbone...")
        # model = timm.create_model('mobilenetv2_100', pretrained=pretrained)
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True) # idk if this works

        # model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=False) # this works

        print(f"The model 'mobilenetv2_100' has {count_model_params(model)} parameters")        

        channels = [160, 96, 32, 24]
        print(f"MobileNetV2 Backbone: channels: {channels}")
        # onyl for debugging

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
    def __init__(self, cfgs = None, pretrained: bool = True, checkpoint_path: str = None):
        super().__init__()
        print("Creating MobileNetV3SmallBackbone...")
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=pretrained)

        print("")
        print("")
        print(f"The model 'mobilenetv3_small_100': {self.model}")
        print("")
        print("")

        print(f"The model 'mobilenetv3_small_100' has {count_model_params(self.model)/1e6}M parameters")

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
    def __init__(self, cfgs = None, pretrained: bool = True, checkpoint_path: str = None):
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
    def __init__(self, cfgs = None, pretrained: bool = True, checkpoint_path: str = None):
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
    def __init__(self, cfgs = None, pretrained: bool = True, checkpoint_path: str = None):
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
    def __init__(self, cfgs = None, pretrained: bool = True, checkpoint_path: str = None):
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
    'LeanBackbone': FeatureExtractionNet,
}

from fvcore.nn import FlopCountAnalysis, parameter_count_table


def build_backbone(backbone: str = 'MobileNetv2', cfgs = None, pretrained: bool = True, checkpoint_path: str = None) -> nn.Module:
    """Single dispatch point to instantiate a backbone class by name."""
    if backbone not in _BACKBONE_REGISTRY:
        raise NotImplementedError(f"Backbone '{backbone}' is not implemented. Available: {list(_BACKBONE_REGISTRY.keys())}")
    model = _BACKBONE_REGISTRY[backbone](cfgs=cfgs, pretrained=pretrained, checkpoint_path=checkpoint_path)
    print(f"The backbone '{backbone}' has {count_model_params(model)/1e6}M parameters")
    
    if hasattr(model, 'output_channels'):
        print(f"Backbone '{backbone}' output channels: {model.output_channels}")
    else:
        print(f"Backbone '{backbone}' has no attribute 'output_channels'.")

    H = 320
    W = 736
    x = torch.randn(1, 3, H, W)

    # Set to eval mode to avoid BatchNorm training requirement
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, x)
        print(f"Backbone '{backbone}' GFLOPs: {flops.total()/1e9}")
        print(parameter_count_table(model))
        raw = flops.by_module()
        # print(f"Backbone '{backbone}' FLOPs by module (fvcore): {raw}")
        pretty = pretty_print_module_counter(raw)  # hotspot modules
        print("Backbone '{backbone}' FLOPs by module (fvcore) - pretty printed:")
        print(pretty)
        print("Done pretty printing FLOPs by module.")

    model.train()  # Restore training mode if needed

    # flops = FlopCountAnalysis(model, x)
    # print(f"Backbone '{backbone}' GFLOPs: {flops.total()/1e9}")
    # print(parameter_count_table(model))
    return model 


class ConcatenationModule(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.should_concat = cfgs.get('CONCAT_LEFT_RIGHT', False) # if not supplied, then assume False
        if self.should_concat:
            self.concat_type = cfgs.get('CONCAT_LEFT_RIGHT_ALONG', None)

            if self.concat_type is None or self.concat_type not in ['batch', 'horizontal', 'vertical', 'multicut']:
                raise ValueError(f"MODEL.BACKBONE_CFGS.CONCAT_LEFT_RIGHT_ALONG parameter must be one of: batch, horizontal, vertical, multicut. Got: {self.concat_type}")
            
            print(f"Concatenation type: {self.concat_type}")

            if self.concat_type == 'multicut':
                self.h_division = cfgs.get('H_DIVISION', 1)
                self.w_division = cfgs.get('W_DIVISION', 2)   
                print(f"Multicut concat: H_DIVISION={self.h_division}, W_DIVISION={self.w_division}")     

        

class ConcatenationAssembly(ConcatenationModule):
    def __init__(self, cfgs):
        super().__init__(cfgs)

        # self.forward = self.passthrough  # default to passthrough if no concatenation

        # if self.should_concat:
        #     if self.concat_type == 'batch':
        #         self.forward = self.assemble_batch
        #     elif self.concat_type == 'horizontal':
        #         self.forward = self.assemble_horizontal
        #     elif self.concat_type == 'vertical':
        #         self.forward = self.assemble_vertical
        #     elif self.concat_type == 'multicut':
        #         self.forward = self.assemble_multicut
        #     else:
        #         raise NotImplementedError(f"Concat type '{self.concat_type}' is not implemented. Available: batch, horizontal, vertical, multicut")

    def forward(self, image_left, image_right):
        if self.should_concat:
            if self.concat_type == 'batch':
                return self.assemble_batch(image_left, image_right)
            elif self.concat_type == 'horizontal':
                return self.assemble_horizontal(image_left, image_right)
            elif self.concat_type == 'vertical':
                return self.assemble_vertical(image_left, image_right)
            elif self.concat_type == 'multicut':
                return self.assemble_multicut(image_left, image_right)
            else:
                raise NotImplementedError(f"Concat type '{self.concat_type}' is not implemented. Available: batch, horizontal, vertical, multicut")
        else:
             return self.passthrough(image_left), self.passthrough(image_right)

    def passthrough(self, image):
        return image

    # concatenate left and right images along batch
    def assemble_batch(self, image_left, image_right):
        return torch.cat([image_left, image_right], dim=0)

    # concatenate along horizontal dimension
    def assemble_horizontal(self, image_left, image_right):
        return torch.cat([image_left, image_right], dim=-1)

    # concatenate along vertical dimension
    def assemble_vertical(self, image_left, image_right):
        return torch.cat([image_left, image_right], dim=-2)

    def assemble_multicut(self, image_left, image_right):
        image_left_parts = []
        image_right_parts = []

        # after division, the new height and width need to be divisible by 32 (depending on the backbone)

        nr_of_parts = self.w_division * self.h_division  # 8
        h, w, b = image_left.size(-2), image_left.size(-1), image_left.size(0)
        h_divided = h // self.h_division
        w_divided = w // self.w_division

        assert h_divided % 32 == 0, f"Divided down height {h_divided} not divisible by 32 (originally {h})"
        assert w_divided % 32 == 0, f"Divided down width {w_divided} not divisible by 32 (originally {w})"
        
        for i in range(self.h_division):
            for j in range(self.w_division):
                image_left_parts.append(image_left[..., i*h_divided:(i+1)*h_divided, j*w_divided:(j+1)*w_divided])
                image_right_parts.append(image_right[..., i*h_divided:(i+1)*h_divided, j*w_divided:(j+1)*w_divided])

        left_stack = torch.cat(image_left_parts, dim=0)
        right_stack = torch.cat(image_right_parts, dim=0)

        # concatenate all parts along batch dimension
        combined = torch.cat((left_stack, right_stack), dim=0)

        return combined
  


class ConcatenationDisassembly(ConcatenationModule):
    def __init__(self, cfgs):
        super().__init__(cfgs)

        # self.forward = self.passthrough  # default to passthrough if no concatenation

        # if self.should_concat:
        #     if self.concat_type == 'batch':
        #         self.forward = self.disassemble_batch
        #     elif self.concat_type == 'horizontal':
        #         self.forward = self.disassemble_horizontal
        #     elif self.concat_type == 'vertical':
        #         self.forward = self.disassemble_vertical
        #     elif self.concat_type == 'multicut':
        #         self.forward = self.disassemble_multicut
        #     else:
        #         raise NotImplementedError(f"Concat type '{self.concat_type}' is not implemented. Available: batch, horizontal, vertical, multicut")

    def forward(self, features):
        if self.should_concat:
            if self.concat_type == 'batch':
                return self.disassemble_batch(features)
            elif self.concat_type == 'horizontal':
                return self.disassemble_horizontal(features)
            elif self.concat_type == 'vertical':
                return self.disassemble_vertical(features)
            elif self.concat_type == 'multicut':
                return self.disassemble_multicut(features)
            else:
                raise NotImplementedError(f"Concat type '{self.concat_type}' is not implemented. Available: batch, horizontal, vertical, multicut")
        else:
             return features
    
    def passthrough(self, features):
        return features

    def disassemble_batch(self, combined_features):
        # split features back to left and right
        initial_batch_size = combined_features[0].size(0) // 2

        features_left = []
        features_right = []

        # split features back to left and right
        for feat in combined_features:            
            features_left.append(feat[:initial_batch_size])
            features_right.append(feat[initial_batch_size:])

        return features_left, features_right

    def disassemble_horizontal(self, combined_features):

        features_left = []
        features_right = []

        # split features back to left and right
        for feat in combined_features:
            w = feat.size(-1)
            w_divided = w // 2
            features_left.append(feat[..., :w_divided])
            features_right.append(feat[..., w_divided:])

        return features_left, features_right

    def disassemble_vertical(self, combined_features):

        features_left = []
        features_right = []

        # split features back to left and right
        for feat in combined_features:
            h = feat.size(-2)
            h_divided = h // 2
            features_left.append(feat[..., :h_divided, :])
            features_right.append(feat[..., h_divided:, :])

        return features_left, features_right

    def disassemble_multicut(self, combined_features):
        features_left = []
        features_right = []

        nr_of_parts = self.w_division * self.h_division

        # h, w, b = image_left.size(-2), image_left.size(-1), image_left.size(0)

        b = combined_features[0].size(0) // (2 * nr_of_parts) # original batch size before multicut and concatenation

        for feat in combined_features:

            # should be [b*2*8, c, h/4, w/2]
            combined_features_left = feat[:nr_of_parts*b, ...]
            combined_features_right = feat[nr_of_parts*b:, ...]

            left_feat_image = []
            right_feat_image = []
            
            for bi in range(b):

                left_v_sequence = []
                right_v_sequence = []
                for i in range(self.h_division): # along height

                    left_h_sequence = []
                    right_h_sequence = []
                    for j in range(self.w_division): # along width

                        left_h_sequence.append(combined_features_left[i*b*self.w_division + j*b + bi, ...]) # append to horizontal sequence
                        right_h_sequence.append(combined_features_right[i*b*self.w_division + j*b + bi, ...])

                    left_v_sequence.append(torch.cat(left_h_sequence, dim=-1)) # first concat along width (get full row), then append to vertical sequence
                    right_v_sequence.append(torch.cat(right_h_sequence, dim=-1))

                left_feat_image.append(torch.cat(left_v_sequence, dim=-2)) # first concat along height (get full image), then append along batch dimension
                right_feat_image.append(torch.cat(right_v_sequence, dim=-2))

            features_left.append(torch.stack(left_feat_image, dim=0)) # first stack along batch, then append to features list
            features_right.append(torch.stack(right_feat_image, dim=0))

        return features_left, features_right



class Backbone(nn.Module):
    """Thin wrapper kept for backward compatibility. Delegates to a specific backbone implementation."""
    def __init__(self, backbone: str = 'MobileNetv2', cfgs = None, pretrained: bool = True, checkpoint_path: str = None):
        super().__init__()
        print(f"Using backbone: {backbone}, pretrained: {pretrained}, checkpoint_path: {checkpoint_path}")
        self.impl = build_backbone(backbone, cfgs=cfgs, pretrained=pretrained, checkpoint_path=checkpoint_path)
        self.assembly = ConcatenationAssembly(cfgs)
        self.disassembly = ConcatenationDisassembly(cfgs)
        self.output_channels = getattr(self.impl, 'output_channels', None)

        # by default, do two separate passes (traditional way)
        # self.forward = self.forward_left_right_one_by_one

        self.should_concat = cfgs.get('CONCAT_LEFT_RIGHT', False)

        # check if batching requested (new proposal)
        # if (cfgs is not None):
        #     if cfgs.get('CONCAT_LEFT_RIGHT', False): # if not supplied, then assume False
        #         self.forward = self.forward_concatenated
    

    def forward(self, image_left, image_right):
        if self.should_concat:
            return self.forward_concatenated(image_left, image_right)
        else:
            return self.forward_left_right_one_by_one(image_left, image_right)

    def forward_left_right_one_by_one(self, image_left, image_right):
   
        # First left
        # image_left = self.assembly(image_left)
        
        # configured as pass through
        image_left, image_right = self.assembly(image_left, image_right)
        
        features_left = self.impl(image_left)
        # features_left = self.disassembly(features_left)

        # Then right
        # image_right = self.assembly(image_right)
        features_right = self.impl(image_right)
        
        features_left = self.disassembly(features_left)
        features_right = self.disassembly(features_right)

        
        
        return features_left, features_right
   
    def forward_concatenated(self, image_left, image_right):

        # Both at the same time
        concatenated_images = self.assembly(image_left, image_right)
        combined_features = self.impl(concatenated_images)
        features_left, features_right = self.disassembly(combined_features)

        return features_left, features_right
    


def count_model_params(model):
    num_params = sum(p.numel() for p in model.parameters())
    return num_params

