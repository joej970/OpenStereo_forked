from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

from tools.debug_utils import debug_printer
# from models.submodule import *
from .submodule import *
import math
# from models.backbone import FeatureExtractionNet
from .backbone import FeatureExtractionNet
from .loss import __loss_type__, model_loss
from .FeatExtractionWrapper import feature_extraction



class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class LeanStereoNet(nn.Module):
    # def __init__(self, cfgs, use_concat_volume=False):
    def __init__(self, cfgs):
        super(LeanStereoNet, self).__init__()
        self.max_disp = cfgs.MAX_DISP
        # self.use_concat_volume = use_concat_volume
        self.use_concat_volume = cfgs.USE_CONCAT_VOLUME
        self.aux_mode = cfgs.aux_mode
        self.loss_type = cfgs.LOSS_TYPE
        self.concat_vol_builder_type = cfgs.get('CONCAT_VOLUME_BUILDER_TYPE', 'original')


        self.backbone_conf = cfgs.get('BACKBONE_CFGS', None)
        
        print(f"LeanStereo aux_mode: {self.aux_mode}, use_concat_volume: {self.use_concat_volume}, loss_type: {self.loss_type}, max_disp: {self.max_disp}, backbone_conf: {self.backbone_conf}")

        # nclasses = cfgs.nclasses
        self.num_groups = int( self.max_disp/ 8)  # BGA layer maps to 128 feat maps

        self.apply_regression= False if cfgs.LOSS_TYPE == "ohemCE" else True

        self.concat_channels = 32
        self.concatconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, self.concat_channels, kernel_size=1, padding=0, stride=1,
                                                  bias=False))
        
        self.concat_volume = concat_volume_builder(builder_type=self.concat_vol_builder_type, maxdisp=self.max_disp // 4)
        # if self.aux_mode == "train":

        self.patch = nn.Conv3d(40, 40, kernel_size=(1, 3, 3), stride=1, dilation=1, groups=40, padding=(0, 1, 1),
                               bias=False)
        self.patch_l1 = nn.Conv3d(8, 8, kernel_size=(1, 3, 3), stride=1, dilation=1, groups=8, padding=(0, 1, 1),
                                  bias=False)
        self.patch_l2 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=1, dilation=2, groups=16, padding=(0, 2, 2),
                                  bias=False)
        self.patch_l3 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=1, dilation=3, groups=16, padding=(0, 3, 3),
                                  bias=False)
        self.dres1_att = nn.Sequential(convbn_3d(40, 16, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn_3d(16, 16, 3, 1, 1))
        self.dres2_att = hourglass(16)
        self.classif_att = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # concat_feature = true reduces number of feature channels to concat_feature_channel
        if self.use_concat_volume:
            # self.concat_channels = 12
            self.feature_extraction = feature_extraction(
                                                        feature_extractor=FeatureExtractionNet(),
                                                         concat_feature=True,
                                                         concat_feature_channel=self.concat_channels, cfgs=self.backbone_conf)
        else:
            # self.concat_channels = 0
            self.feature_extraction = feature_extraction(
                feature_extractor=FeatureExtractionNet(), concat_feature=False, cfgs=self.backbone_conf)

        self.dres0 = nn.Sequential(convbn_3d(self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # nn.init.normal(self.sum_weights, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def build_acvnet_volume(self, features_left, features_right):
        gwc_volume1 = build_gwc_volume(features_left, features_right, self.max_disp // 4,
                                       num_groups=40)
        gwc_volume = self.patch(gwc_volume1)  # >>>>>>> Gwc-p <<<<<<<<<<<<
        patch_l1 = self.patch_l1(gwc_volume[:, :8])
        patch_l2 = self.patch_l2(gwc_volume[:, 8:24])
        patch_l3 = self.patch_l3(gwc_volume[:, 24:40])
        patch_volume = torch.cat((patch_l1, patch_l2, patch_l3), dim=1)  # >>>>>>> Gwc-mp <<<<<<<<<<<<
        cost_attention = self.dres1_att(patch_volume)
        cost_attention = self.dres2_att(cost_attention)  # >>>>>>> Gwc-mp-hg <<<<<<<<<<<<
        att_weights = self.classif_att(cost_attention)
        return att_weights

    # def forward(self, left, right):
    def forward(self, data):

        left = data['left']
        right = data['right']

        features_left, features_right = self.feature_extraction(left, right) # [H/8, W/8]


        concat_feature_left = self.concatconv(features_left["features"]) # conv + bn + relu + conv
        concat_feature_right = self.concatconv(features_right["features"])
        gwc_volume = self.concat_volume(concat_feature_left, concat_feature_right)

        att_weights = self.build_acvnet_volume(features_left["features"], features_right["features"])

        s_max = F.softmax(att_weights, dim=2)

        volume = gwc_volume.mul_(s_max)

        cost0 = self.dres0(volume) # seq 1 # right before Mul before ScatterND_95 from 300 experiment 
        cost0 = self.dres1(cost0) + cost0 # seq 2

        out1 = self.dres2(cost0) # hour glass 1 
        out2 = self.dres3(out1) # hour glass 2

        if self.aux_mode == "train":
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)

            cost0 = F.interpolate(cost0, [self.max_disp, left.size()[2], left.size()[3]], mode='trilinear')
            cost0 = cost0[:, 0, ...]

            cost1 = F.interpolate(cost1, [self.max_disp, left.size()[2], left.size()[3]], mode='trilinear')
            cost1 = cost1[:, 0, ...]

            cost2 = F.interpolate(cost2, [self.max_disp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = cost2[:, 0, ...]

            if self.apply_regression:
                pred0 = F.softmax(cost0, dim=1)
                pred0 = disparity_regression(pred0, self.max_disp)

                pred1 = F.softmax(cost1, dim=1)
                pred1 = disparity_regression(pred1, self.max_disp)

                pred2 = F.softmax(cost2, dim=1)
                # print(f"train: Softmaxed cost2 size: {pred2.size()}")
                pred2 = disparity_regression(pred2, self.max_disp)
                # print(f"train: Final predicted disparity size: {pred2.size()}")
            else:
                pred0, pred1, pred2 = cost0, cost1, cost2
        
            return {"disparities":[pred0, pred1, pred2], "disp_pred": pred2}

        else:
            
            cost2 = self.classif2(out2)
            # test/eval: Raw cost2 size: torch.Size([1, 1, 48, 68, 120])
            
            cost2 = F.upsample(cost2, [self.max_disp, left.size()[2], left.size()[3]], mode='trilinear')
            # test/eval: Upsampled cost2 size: torch.Size([1, 1, 192, 544, 960])
            
            cost2 = cost2[:, 0, ...]
            # test/eval: cost2 size: torch.Size([1, 192, 544, 960])
            
            pred2 = F.softmax(cost2, dim=1)
            # test/eval: Softmaxed cost2 size: torch.Size([1, 192, 544, 960])
            
            pred2 = disparity_regression(pred2, self.max_disp)
            # test/eval: Final predicted disparity size: torch.Size([1, 544, 960])
            
            # return [pred2]
            return {"disparities":[pred2], "disp_pred": pred2}

        
    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        # disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]

        loss = model_loss(model_pred, disp_gt, mask, __loss_type__[self.loss_type])

        if torch.isnan(loss):
            print(f"NaN loss detected at batch {input_data['iteration']} (loss)", end="")   
            print(f"Input left contains NaN: {torch.isnan(input_data['left']).any()}", end="")
            print(f"Input rigth contains NaN: {torch.isnan(input_data['right']).any()}", end="")
            print(f"Target contains NaN: {torch.isnan(input_data['disp']).any()}")
        

        loss_info = {'scalar/train/loss_disp': loss.item()}
        return loss, loss_info
        

# def LeanStereo(cfgs):
#     return LeanStereoNet(cfgs, use_concat_volume=False)
