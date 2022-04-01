import torch
import torch.nn as nn
from .Loss import Loss_loc_cls


class BoundaryAdjust(nn.Module):

    def __init__(self, opt):
        super(BoundaryAdjust, self).__init__()
        self.tscale = opt['temporal_scale']
        self.base_stride = opt['base_stride']
        bb_hidden_dim = opt['bb_hidden_dim']

        self.Loss = Loss_loc_cls(opt)

        self.start_conv = nn.Sequential(
            nn.Conv1d(in_channels=bb_hidden_dim, out_channels=bb_hidden_dim,kernel_size=3,stride=2,padding=0,groups=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=bb_hidden_dim, out_channels=1, kernel_size=1),
        )

        self.end_conv = nn.Sequential(
            nn.Conv1d(in_channels=bb_hidden_dim, out_channels=bb_hidden_dim,kernel_size=3,stride=2,padding=0,groups=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=bb_hidden_dim, out_channels=1, kernel_size=1),
        )

    def forward(self, loc_box, feat_frmlvl):
        beta = 8.
        bs, C, _ = feat_frmlvl.shape

        if isinstance(loc_box, list):
            loc_box = torch.cat(loc_box, dim=1)
        loc_box = torch.clamp(loc_box, min=0., max=(self.tscale)-1)

        boundary_length = (loc_box[:,:,1] - loc_box[:,:,0] + 1) / beta

        # Starts
        starts = torch.clamp((loc_box[:,:,0] / self.base_stride).to(dtype=torch.long), min=0, max=(self.tscale / self.base_stride)-1)
        starts_left = torch.clamp(((loc_box[:,:,0] - boundary_length) / self.base_stride).to(dtype=torch.long), min=0, max=(self.tscale / self.base_stride)-1)
        starts_right = torch.clamp(((loc_box[:,:,0] + boundary_length) / self.base_stride).to(dtype=torch.long), min=0, max=(self.tscale / self.base_stride)-1)

        start_feats_center = torch.stack([feat_frmlvl[i, :, starts[i, :]] for i in range(bs)]).permute(0, 2, 1).reshape(-1, C)
        start_feats_left = torch.stack([feat_frmlvl[i, :, starts_left[i, :]] for i in range(bs)]).permute(0, 2, 1).reshape(-1, C)
        start_feats_right = torch.stack([feat_frmlvl[i, :, starts_right[i, :]] for i in range(bs)]).permute(0, 2, 1).reshape(-1, C)

        start_feats = torch.stack((start_feats_left, start_feats_center, start_feats_right), dim=-1)

        start_offsets = self.start_conv(start_feats).squeeze().view(bs,-1)

        # Ends
        ends = torch.clamp((loc_box[:,:,1] / self.base_stride).to(dtype=torch.long), min=0, max=(self.tscale / self.base_stride)-1)
        ends_left = torch.clamp(((loc_box[:,:,1] - boundary_length) / self.base_stride).to(dtype=torch.long), min=0, max=(self.tscale / self.base_stride)-1)
        ends_right = torch.clamp(((loc_box[:,:,1] + boundary_length) / self.base_stride).to(dtype=torch.long), min=0, max=(self.tscale / self.base_stride)-1)

        end_feats_center = torch.stack([feat_frmlvl[i, :, ends[i, :]] for i in range(bs)]).permute(0, 2, 1).reshape(-1, C)
        end_feats_left = torch.stack([feat_frmlvl[i, :, ends_left[i, :]] for i in range(bs)]).permute(0, 2, 1).reshape(-1, C)
        end_feats_right = torch.stack([feat_frmlvl[i, :, ends_right[i, :]] for i in range(bs)]).permute(0, 2, 1).reshape(-1, C)

        end_feats = torch.stack((end_feats_left, end_feats_center, end_feats_right), dim=-1)

        end_offsets = self.end_conv(end_feats).squeeze().view(bs,-1)

        return start_offsets, end_offsets

    def cal_loss(self, start_offsets, end_offsets, anchors, gt_bbox, num_gt):

        box_pred = self.update_bd(anchors, start_offsets, end_offsets).view(-1, 2)

        cls_labels, reg_targets = self.Loss.prepare_targets(gt_bbox, num_gt, anchors, stage=2)

        cls_labels = torch.cat(cls_labels, dim=0)  # bs*levels*positions*scales, num_cls
        reg_targets = torch.cat(reg_targets, dim=0)  # bs*levels*positions*scales, left-right

        all_anchors = torch.cat(anchors, dim=1).view(-1, 2) # bs*levels*positions*scales, 2

        pos_inds = torch.nonzero(cls_labels > 0).squeeze(1)
        reg_loss = self.Loss.reg_loss_func(None, reg_targets[pos_inds], all_anchors[pos_inds], box_pred[pos_inds]) / pos_inds.numel()

        return reg_loss

    def update_bd(self, anchors, start_offsets, end_offsets):

        if isinstance(anchors, list):
            anchors = torch.cat(anchors, dim=1)
        loc_st2 = anchors.clone()
        loc_st2[:,:,0] = anchors[:,:,0] + start_offsets
        loc_st2[:,:,1] = anchors[:,:,1] + end_offsets

        return loc_st2
