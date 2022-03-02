# -*- coding: utf-8 -*-
import torch.nn as nn
from .XGPN import XGPN
import torch.nn.functional as F
from .Head import Head
from .AnchorGenerator import AnchorGenerator
from .Loss import Loss_loc_cls, get_loss_supplement
from .ActionGenerator import Pred_loc_cls
from .BoundaryAdjust import BoundaryAdjust

class VSGN(nn.Module):
    def __init__(self, opt):
        super(VSGN, self).__init__()

        self.bb_hidden_dim = opt['bb_hidden_dim']
        self.bs = opt["batch_size"]
        self.is_train = opt['is_train']
        self.tem_best_loss = 10000000
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512
        self.input_feat_dim = opt['input_feat_dim']

        self.xGPN = XGPN(opt)

        # self.head_enc = Head(opt)
        self.head_dec = Head(opt)

        self.anchors = AnchorGenerator(opt).anchors
        self.loss_loc_and_cls = Loss_loc_cls(opt)
        self.bd_adjust = BoundaryAdjust(opt)

        self.pred_loc_and_cls = Pred_loc_cls(opt)


        # Generate action/start/end scores
        self.head_actionness = nn.Sequential(
            nn.Conv1d(self.bb_hidden_dim, self.bb_hidden_dim, kernel_size=3, padding=1, groups=1),
            # nn.GroupNorm(32, self.bb_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.bb_hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.head_startness = nn.Sequential(
            nn.Conv1d(self.bb_hidden_dim, self.bb_hidden_dim, kernel_size=3, padding=1, groups=1),
            # nn.GroupNorm(32, self.bb_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.bb_hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.head_endness = nn.Sequential(
            nn.Conv1d(self.bb_hidden_dim, self.bb_hidden_dim, kernel_size=3, padding=1, groups=1),
            # nn.GroupNorm(32, self.bb_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.bb_hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def _forward_test(self, cls_pred_enc, reg_pred_enc, cls_pred_dec, reg_pred_dec):

        loc_enc, score_enc, loc_dec, score_dec = self.gen_predictions(cls_pred_enc, reg_pred_enc, cls_pred_dec, reg_pred_dec, self.anchors)

        return loc_enc, score_enc, loc_dec, score_dec



    def forward(self, input, num_frms, gt_action= None, gt_start= None, gt_end= None, gt_bbox = None, num_gt = None):
        # Cross-scale graph pyramid network
        feats_enc, feats_dec = self.xGPN(input, num_frms)

        # Scoring and localization
        cls_pred_dec, reg_pred_dec = self.head_dec(feats_dec)
        if self.is_train == 'true':
            losses, loc_dec = self.loss_loc_and_cls(cls_pred_dec, reg_pred_dec, self.anchors, gt_bbox, num_gt)
        else:
            score_dec, loc_dec = self.pred_loc_and_cls(cls_pred_dec, reg_pred_dec, self.anchors)

        # Supplementary scores
        actionness = self.head_actionness(feats_dec[-1])
        startness = self.head_startness(feats_dec[-1])
        endness = self.head_endness(feats_dec[-1])

        actionness = F.interpolate(actionness, size=input.size()[2:], mode='linear', align_corners=True).squeeze(1)
        startness = F.interpolate(startness, size=input.size()[2:], mode='linear', align_corners=True).squeeze(1)
        endness = F.interpolate(endness, size=input.size()[2:], mode='linear', align_corners=True).squeeze(1)

        if self.is_train == 'true':
            loss_action, loss_start, loss_end = get_loss_supplement(actionness, gt_action, startness, gt_start, endness, gt_end)
            losses['loss_action'] = loss_action
            losses['loss_start'] = loss_start
            losses['loss_end'] = loss_end

        # Boundary adjustment
        start_offsets, end_offsets = self.bd_adjust(loc_dec, feats_dec[-1])

        if self.is_train == 'true':
            loss_reg_st2 = self.bd_adjust.cal_loss(start_offsets, end_offsets, loc_dec, gt_bbox, num_gt)
            losses['loss_bd_adjust'] = loss_reg_st2

            return losses, actionness, startness, endness
        else:
            loc_adjusted = self.bd_adjust.update_bd(loc_dec, start_offsets, end_offsets)
            return loc_dec, score_dec, loc_adjusted, actionness, startness, endness






