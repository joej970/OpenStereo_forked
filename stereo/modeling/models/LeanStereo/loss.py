import torch.nn.functional as F
import torch.nn as nn
import torch
from functools import partial



def model_loss(disp_ests, disp_gt, mask, loss_func=None):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        # print(f"disp_est shape: {disp_est.shape}, disp_gt shape: {disp_gt.shape}, mask shape: {mask.shape}")
        # disp_est shape: torch.Size([16, 320, 736]), disp_gt shape: torch.Size([16, 320, 736]), mask shape: torch.Size([16, 320, 736])
        # disp_est shape: torch.Size([16, 320, 736]), disp_gt shape: torch.Size([16, 320, 736]), mask shape: torch.Size([16, 320, 736])
        # disp_est shape: torch.Size([16, 320, 736]), disp_gt shape: torch.Size([16, 320, 736]), mask shape: torch.Size([16, 320, 736])
        all_losses.append(weight * loss_func(disp_est[mask], disp_gt[mask]))
    return sum(all_losses)

# class OhemCELoss(nn.Module):
#
#     def __init__(self, thresh=0.7, ignore_index=193):
#         super(OhemCELoss, self).__init__()
#         self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
#         # self.ignore_lb = ignore_lb
#         self.criteria = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
#
#     def forward(self, logits, labels):
#         n_min = labels.numel() // 16
#         loss = self.criteria(logits, labels).view(-1)
#         loss_hard = loss[loss > self.thresh]
#         if loss_hard.numel() < n_min:
#             loss_hard, _ = loss.topk(n_min)
#         return torch.mean(loss_hard)

class LogL1Loss(nn.Module):
    """
    Loss to apply at student vs teacher predictions and/or student vs GT (replacing smoothL1)
    """
    def __init__(self):
        super(LogL1Loss, self).__init__()
        self.crit = nn.L1Loss()

    def forward(self, f_s, f_t):
        f_s= torch.log(f_s + 0.5)
        f_t= torch.log(f_t + 0.5)
        loss = self.crit(f_s, f_t)
        return loss


class LogL1Loss_v2(nn.Module):
    """
    Loss to apply at student vs teacher predictions and/or student vs GT (replacing smoothL1)
    """
    def __init__(self):
        super(LogL1Loss_v2, self).__init__()
        self.crit = nn.L1Loss(reduction=None)

    def forward(self, f_s, f_t):
        loss = torch.log(torch.abs(f_s - f_t) + 0.5).mean()
        return loss
    

__loss_type__ = {
    "smoothL1": partial(F.smooth_l1_loss, reduction="mean"),
    "huber": partial(F.huber_loss),
    "l2": partial(F.mse_loss),
    "LogL1Loss": LogL1Loss(),
    "LogL1Loss_v2": LogL1Loss_v2()
    # "ohemCE": OhemCELoss()
}