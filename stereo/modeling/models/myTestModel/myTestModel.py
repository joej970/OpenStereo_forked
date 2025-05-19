import torch
import torch.nn as nn


class myTestModel(torch.nn.Module):
    def __init__(self, cfgs, args):
        super(myTestModel, self).__init__()
        self.cfgs = cfgs
        self.args = args

    def forward(self, x):
        # return x + 1
    
        # Feature extraction

        # Matching cost computation

        # Cost aggregation

        # Refinement
        
        # Disparity regression

        # Output the final disparity map

        init_disp = 0
        disp_preds = []
        return {'init_disp': init_disp,
            'disp_preds': disp_preds,
            'disp_pred': disp_preds[-1]}

    def get_loss(self, pred, target):
        # Compute the loss between the predicted and target disparity maps
        loss = nn.MSELoss()(pred, target)
        return loss
    
    def get_loss(self, model_preds, input_data):
        disp_gt = input_data["disp"] 
        disp_pred = model_preds['disp_pred']