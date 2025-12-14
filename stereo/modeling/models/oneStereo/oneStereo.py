import torch
import torch.nn as nn


import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
# from stereo.modeling.cost_volume.cost_volume import correlation_volume
import stereo.modeling.cost_volume.cost_volume as cost_volumes
from stereo.modeling.disp_pred.disp_regression import disparity_regression
from stereo.modeling.disp_refinement.disp_refinement import context_upsample

from .backbone import Backbone, FPNLayer
from .aggregation import Aggregation, BasicDepthEnrichment, AggregationLittle
from .fusion import FuseDepth
from tools.debug_utils import debug_printer

# is derived from lightstereo model
class oneStereo(torch.nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP
        self.left_att = cfgs.LEFT_ATT

        self.backbone_conf = cfgs.get('BACKBONE_CFGS', None)
        # backbone
        self.backbone = Backbone(backbone=cfgs.get('BACKBONE', 'MobileNetv2'), cfgs=self.backbone_conf, pretrained=cfgs.get('BACKBONE_PRETRAINED', None), checkpoint_path=cfgs.get('BACKBONE_PATH', None))
        # self.backbone = Backbone(cfgs.get('BACKBONE', 'mobileone_s0'))
        self.gated_branch = None
        self.zero_initialised_conv = None

        self.additional_depth_src = cfgs.get('ADDITIONAL_DEPTH_SRC', False)
        if self.additional_depth_src:
            out_chs = [self.max_disp // 4, self.max_disp // 4, self.max_disp // 8] # 48, 48, 24
            # self.twoD_depth_enrichment = BasicDepthEnrichment(intermediate_channels=[48, 32, 16], out_depth_bins=out_chs, mlp_hidden_dim=128)
            intermediate_channels = cfgs.get('BASIC_DEPTH_ENRICHEMENT_CHANNELS', [48, 32, 16])
            self.twoD_depth_enrichment = BasicDepthEnrichment(intermediate_channels=intermediate_channels, out_depth_bins=out_chs, mlp_hidden_dim=128)

            self.depth_src_aggregator = AggregationLittle(depth_channels_in = out_chs[1:], left_att=self.left_att, blocks = [1, 2], expanse_ratio = cfgs.EXPANSE_RATIO_DEPTH_SOURCE_AGGREGATION, backbone_channels=self.backbone.output_channels)

            # option 1: just normal 1x1 conv with zero init
            if cfgs.get('ZERO_INITIALISED_CONV_BEFORE_FUSION', False):
                self.zero_initialised_conv = nn.Conv2d(out_chs[0], out_chs[0], 1, bias=True)
                nn.init.zeros_(self.zero_initialised_conv.weight)
                nn.init.zeros_(self.zero_initialised_conv.bias)

            # option 2: learns how much to add
            if cfgs.get('GATED_BRANCH_BEFORE_FUSION', False):
                self.gated_branch = GatedBranch()

            self.fusion = FuseDepth(cfgs.FUSION_TYPE, out_chs[0], self.max_disp // 4, out_depth_bins = self.max_disp // 4, num_convs = cfgs.FUSION_CONVS, expanse_ratio=cfgs.EXPANSE_RATIO_FUSION, ignore_add_depth_debug_only=cfgs.get('IGNORE_ADD_DEPTH_DEBUG_ONLY', False))


        COST_VOL_TYPE = cfgs.get('COST_VOLUME_TYPE', 'SIMPLE_3D')
        self.cost_volume = cost_volumes.SimpleCorrelationVolumes(type = COST_VOL_TYPE)

        cost_vol_is_4d = COST_VOL_TYPE[-2:] == '4D'
        if cost_vol_is_4d:
            print(f"backbone out channels: {self.backbone.output_channels}")
            input_channels = self.backbone.output_channels[0]
            hidden_dim = [4*input_channels, self.max_disp, 1]
            self.infer_3d = cost_volumes.Infer3DbyMLP(input_channels, hidden_dim, activation='relu', dropout=0.0, normalization = 'layer')
        else:
            self.infer_3d = None

        # aggregation
        self.cost_agg = Aggregation(in_channels=self.max_disp // 4,
                                    left_att=self.left_att,
                                    blocks=cfgs.AGGREGATION_BLOCKS,
                                    expanse_ratio=cfgs.EXPANSE_RATIO,
                                    backbone_channels=self.backbone.output_channels)

        # disp refine
        self.refine_1 = nn.Sequential(
            BasicConv2d(self.backbone.output_channels[0], 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU))

        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.refine_2 = FPNLayer(24, 16)

        self.refine_3 = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)

        # self.half()  # use half precision for memory saving
        # print()
        # print()
        # print("oneStereo model initialized in .half() configuration. input data is also .half()!!!")
        # print("oneStereo model initialized in .half() configuration. input data is also .half()!!!")
        # print()
        # print("oneStereo model initialized in .half() configuration. input data is also .half()!!!")
        # print()
        # print("oneStereo model initialized in .half() configuration. input data is also .half()!!!")
        # print("oneStereo model initialized in .half() configuration. input data is also .half()!!!")
        # print()
        # print()


    def forward(self, data):

        # return x + 1
    
        # Feature extraction

        # Matching cost computation

        # Cost aggregation

        # Refinement
        
        # Disparity regression

        # Output the final disparity map

        debug_printer.print_of_function(lambda: f"----------------------------------- Forward Pass -------------------")

        image1 = data['left'] # [24, 3, 544, 960], tensor contigous = yes type: <class 'torch.Tensor'>
        image2 = data['right']

        # image1 = data['left'].half() # does not work with leanbackbone (NaN issues)
        # image2 = data['right'].half()

        # print(f"Input size: image1: {image1.size()}")

        debug_printer.print_of_function(lambda : f"Input image1 stats: min={image1.min():.3f}, max={image1.max():.3f}, mean={image1.mean():.3f}, std={image1.std():.3f}")
    # Features left sizes: [torch.Size([24, 16, 80, 184]), torch.Size([24, 24, 40, 92]), torch.Size([24, 40, 20, 46]), torch.Size([24, 96, 10, 23])]

        if self.backbone_conf.get('CONCAT_LEFT_RIGHT', False):
            features_left, features_right = self.backbone(image1, image2) # [H/4, W/4 | H/8, W/8 | H/16, W/16 | H/32, W/32]
        else:
            features_left = self.backbone(image1) # [H/4, W/4 | H/8, W/8 | H/16, W/16 | H/32, W/32]
            # for right we only need at H/4, W/4 but at for enc-dec we have to calculate all anyways 
            features_right = self.backbone(image2) 

        # print(f"Features left sizes: {[f.size() for f in features_left]}")

        # debug_printer.print_of_function(lambda : f"Backbone output feature[0] left stats: min={features_left[0].min():.3f}, max={features_left[0].max():.3f}, mean={features_left[0].mean():.3f}, std={features_left[0].std():.3f}")

        # debug_printer.print_of_function(lambda : f"Backbone output feature[1] left stats: min={features_left[1].min():.3f}, max={features_left[1].max():.3f}, mean={features_left[1].mean():.3f}, std={features_left[1].std():.3f}")
    
        # print type and shape of features
        # debug_printer.print_of_function(lambda : f"Backbone output feature: type: {[f.type() for f in features_left]}, shape: {[f.shape for f in features_left]}")


        #  MobileNetV2  (exp 40) [train: Epoch 0, Sample 62 ] Backbone output feature: type: ['torch.cuda.HalfTensor', 'torch.cuda.HalfTensor', 'torch.cuda.HalfTensor', 'torch.cuda.HalfTensor'], shape: [torch.Size([24, 24, 80, 184]), torch.Size([24, 32, 40, 92]), torch.Size([24, 96, 20, 46]), torch.Size([24, 160, 10, 23])]

        # LeanBackbone (exp 211) [train: Epoch 0, Sample 62 ] Backbone output feature: type: ['torch.cuda.FloatTensor', 'torch.cuda.FloatTensor', 'torch.cuda.FloatTensor'], shape: [torch.Size([24, 24, 80, 184]), torch.Size([24, 32, 40, 92]), torch.Size([24, 96, 20, 46])]


        # todo: maybe features[0] at H/4 resolution is not detailed enough?
        gwc_volume = self.cost_volume.calculate(features_left[0], features_right[0], self.max_disp // 4)

        if self.infer_3d is not None: # learn the weighting function on your own
            gwc_volume = self.infer_3d(gwc_volume)

        encoding_volume = self.cost_agg(gwc_volume, features_left)  

        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"Cost aggregation: encoding_volume[0] Found NaN!", force_print=encoding_volume[0].isfinite().all()==False)

        debug_printer.print_of_function(lambda : f"stereo: encoding_volume[0] stats: min={encoding_volume[0].min():.3f}, max={encoding_volume[0].max():.3f}, mean={encoding_volume[0].mean():.3f}, std={encoding_volume[0].std():.3f}")

        if self.additional_depth_src:
            depth_source = data["depth_src_0"]  # [bz, h, w]

            debug_printer.print_of_function(lambda : f"depth src: depth_source stats: min={depth_source.min():.3f}, max={depth_source.max():.3f}, mean={depth_source.mean():.3f}, std={depth_source.std():.3f}")
           
            depth_source = depth_source.unsqueeze(1)  # [bz, 1, h, w]

            depth_enriched = self.twoD_depth_enrichment(depth_source)  # [self.max_disp // 4, self.max_disp // 8]
            # print(f"Depth enriched[0] stats: min={depth_enriched[0].min():.3f}, max={depth_enriched[0].max():.3f}, mean={depth_enriched[0].mean():.3f}, std={depth_enriched[0].std():.3f}")

            depth_aggregated = self.depth_src_aggregator(depth_enriched[1:], features_left)  # [bz, 1, h, w]

            # debug_printer.print(f"Before fusion - encoding_volume[0] stats: min={encoding_volume[0].min():.3f}, max={encoding_volume[0].max():.3f}, mean={encoding_volume[0].mean():.3f}, std={encoding_volume[0].std():.3f}")
            # debug_printer.print(f"Before fusion - depth_aggregated[0] stats: min={depth_aggregated[0].min():.3f}, max={depth_aggregated[0].max():.3f}, mean={depth_aggregated[0].mean():.3f}, std={depth_aggregated[0].std():.3f}")

            debug_printer.print_of_function(lambda : f"depth src: depth_aggregated[0] stats: min={depth_aggregated[0].min():.3f}, max={depth_aggregated[0].max():.3f}, mean={depth_aggregated[0].mean():.3f}, std={depth_aggregated[0].std():.3f}")

            # option 1
            if self.zero_initialised_conv is not None:
                depth_aggregated[0] = self.zero_initialised_conv(depth_aggregated[0])

            # option 2
            if self.gated_branch is not None:
                depth_aggregated[0] = self.gated_branch(depth_aggregated[0])
                debug_printer.print_of_function(lambda : f"Gated fusion: alpha: {self.gated_branch.get_alpha():.4f}")

            encoding_volume = self.fusion(encoding_volume[0], depth_aggregated[0])
            # encoding_volume = self.fusion(encoding_volume_norm, depth_aggregated_norm)

            # print(f"After fusion - encoding_volume stats: min={encoding_volume[0].min():.3f}, max={encoding_volume[0].max():.3f}, mean={encoding_volume[0].mean():.3f}, std = {encoding_volume[0].std():.3f}")

            debug_printer.print_of_function(lambda : f"After fusion - encoding_volume stats: min={encoding_volume[0].min():.3f}, max={encoding_volume[0].max():.3f}, mean={encoding_volume[0].mean():.3f}, std = {encoding_volume[0].std():.3f}")

        squeezed_encoding = encoding_volume[0].reshape(encoding_volume[0].size(0), -1, encoding_volume[0].size(2), encoding_volume[0].size(3))  # [bz, max_disp/4, H/4, W/4]

        # raise NameError("Force stop in this non-depth variant")

        prob = F.softmax(squeezed_encoding, dim=1)
        init_disp = disparity_regression(prob, self.max_disp // 4)  # [bz, 1, H/4, W/4]

        debug_printer.print_of_function(lambda : f"Initial disparity (disparity_regression) stats: min={init_disp.min():.3f}, max={init_disp.max():.3f}, mean={init_disp.mean():.3f}, std={init_disp.std():.3f}")

        xspx = self.refine_1(features_left[0])
        try:
            xspx = self.refine_2(xspx, self.stem_2(image1)) 
            # the problem: xspx is size 40 (upsampled to 80) while stem_2() is 160
        except Exception as e:
            print(f"Error in refine_2: {e}")
            print(f"xspx shape: {xspx.shape}, stem_2(image1) shape: {self.stem_2(image1).shape}")
            raise e

        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)  # [bz, 9, H, W]
        disp_pred = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # # [bz, 1, H, W]

        if self.training:
            debug_printer.print_of_function_force_print(lambda : f"Refined disparity (context_upsample) Found NaN!", force_print = disp_pred.isfinite().all() == False)

        debug_printer.print_of_function(lambda : f"spx_pred stats: min={spx_pred.min():.3f}, max={spx_pred.max():.3f}, mean={spx_pred.mean():.3f}, std={spx_pred.std():.3f}")

        result = {'disp_pred': disp_pred}

        if self.training:
            disp_4 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
            disp_4 *= 4
            result['disp_4'] = disp_4

        return result

    def get_loss(self, model_pred, input_data):

        # if model_pred is halfTensor, convert to floatTensor for loss calculation
        if model_pred['disp_pred'].dtype == torch.half:
            model_pred['disp_pred'] = model_pred['disp_pred'].float()
        if 'disp_4' in model_pred and model_pred['disp_4'].dtype == torch.half:
            model_pred['disp_4'] = model_pred['disp_4'].float()
            

        debug_printer.print_of_function_force_print(lambda: f"get_loss(): disp_pred Found NaN!", force_print=model_pred['disp_pred'].isfinite().all() == False)
        debug_printer.print_of_function_force_print(lambda: f"get_loss(): input disp Found NaN!" , force_print=input_data['disp'].isfinite().all() == False)

        disp_gt = input_data["disp"]  # [bz, h, w]
        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]

        disp_pred = model_pred['disp_pred']
        disp_4 = model_pred['disp_4']

        # DEBUGGING: Check for problematic values
        # print(f"Mask sum: {mask.sum().item()}")  # Check if mask is empty
        
        if mask.sum() == 0:
            print("WARNING: Empty mask! All pixels filtered out.")
            return torch.tensor(0.0, device=disp_pred.device), {}
        
        # Check for NaN/Inf in predictions
        if torch.isnan(disp_pred).any():
            print(f"NaN in disp_pred: {torch.isnan(disp_pred).sum().item()} pixels")
        if torch.isinf(disp_pred).any():
            print(f"Inf in disp_pred: {torch.isinf(disp_pred).sum().item()} pixels")
        if torch.isnan(disp_4).any():
            print(f"NaN in disp_4: {torch.isnan(disp_4).sum().item()} pixels")
        if torch.isinf(disp_4).any():
            print(f"Inf in disp_4: {torch.isinf(disp_4).sum().item()} pixels")
        
        # Check for NaN/Inf in ground truth
        if torch.isnan(disp_gt).any():
            print(f"NaN in disp_gt: {torch.isnan(disp_gt).sum().item()} pixels")

        loss_1 = 1.0 * F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')

        # loss += 0.3 * F.smooth_l1_loss(disp_4[mask], disp_gt[mask], reduction='mean')
        loss_2 = 0.3 * F.smooth_l1_loss(disp_4[mask], disp_gt[mask], reduction='mean')

        loss = loss_1 + loss_2
        if torch.isnan(loss_1):
            print(f"NaN loss detected at batch {input_data['iteration']} (loss_1)", end="")   
            print(f"Input left contains NaN: {torch.isnan(input_data['left']).any()}", end="")
            print(f"Input rigth contains NaN: {torch.isnan(input_data['right']).any()}", end="")
            print(f"Target contains NaN: {torch.isnan(input_data['disp']).any()}")

        if torch.isnan(loss_2):
            print(f"NaN loss detected at batch {input_data['iteration']} (loss_2)", end="")
            print(f"Input left contains NaN: {torch.isnan(input_data['left']).any()}", end="")
            print(f"Input rigth contains NaN: {torch.isnan(input_data['right']).any()}", end="")
            print(f"Target contains NaN: {torch.isnan(input_data['disp']).any()}")
        

        loss_info = {'scalar/train/loss_disp': loss.item()}

        return loss, loss_info


class GatedBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # start mid open
    def get_alpha(self):
        return self.alpha.item()
    def forward(self, x):
        return self.alpha * x