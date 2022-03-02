
import torch
import math

class BoxCoder(object):

    def __init__(self, opt):
        self.cfg = opt

    def encode(self, gt_boxes, anchors):
        if False: #self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'POINT':
            TO_REMOVE = 1  # TODO remove
            anchors_w = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            anchors_h = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
            anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2

            w = self.cfg.MODEL.ATSS.ANCHOR_SIZES[0] / self.cfg.MODEL.ATSS.ANCHOR_STRIDES[0]
            l = w * (anchors_cx - gt_boxes[:, 0]) / anchors_w
            t = w * (anchors_cy - gt_boxes[:, 1]) / anchors_h
            r = w * (gt_boxes[:, 2] - anchors_cx) / anchors_w
            b = w * (gt_boxes[:, 3] - anchors_cy) / anchors_h
            targets = torch.stack([l, t, r, b], dim=1)
        elif True: #self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'BOX':
            TO_REMOVE = 1  # TODO remove
            ex_length = anchors[:, 1] - anchors[:, 0] + TO_REMOVE
            ex_center = (anchors[:, 1] + anchors[:, 0]) / 2

            gt_length = gt_boxes[:, 1] - gt_boxes[:, 0] + TO_REMOVE
            gt_center = (gt_boxes[:, 1] + gt_boxes[:, 0]) / 2

            wx, ww = (10., 5.)
            targets_dx = wx * (gt_center - ex_center) / ex_length
            targets_dw = ww * torch.log(gt_length / ex_length)
            targets = torch.stack((targets_dx, targets_dw), dim=1)

        return targets

    def decode(self, preds, anchors):
        if False: #self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'POINT':
            TO_REMOVE = 1  # TODO remove
            anchors_w = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            anchors_h = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
            anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2

            w = self.cfg.MODEL.ATSS.ANCHOR_SIZES[0] / self.cfg.MODEL.ATSS.ANCHOR_STRIDES[0]
            x1 = anchors_cx - preds[:, 0] / w * anchors_w
            y1 = anchors_cy - preds[:, 1] / w * anchors_h
            x2 = anchors_cx + preds[:, 2] / w * anchors_w
            y2 = anchors_cy + preds[:, 3] / w * anchors_h
            pred_boxes = torch.stack([x1, y1, x2, y2], dim=1)
        elif True: #self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'BOX':
            anchors = anchors.to(preds.dtype)

            TO_REMOVE = 1  # TODO remove
            ex_length = anchors[:, 1] - anchors[:, 0] + TO_REMOVE
            ex_center = (anchors[:, 1] + anchors[:, 0]) / 2

            wx, ww = (10, 5.)
            dx = preds[:, 0] / wx
            dw = preds[:, 1] / ww

            # Prevent sending too large values into torch.exp()
            dw = torch.clamp(dw, max=math.log(1000. / 16))

            pred_ctr_x = (dx * ex_length + ex_center)
            pred_w = (torch.exp(dw) * ex_length)

            pred_boxes = torch.zeros_like(preds)
            pred_boxes[:, 0] = pred_ctr_x - 0.5 * (pred_w - 1)
            pred_boxes[:, 1] = pred_ctr_x + 0.5 * (pred_w - 1)
        return pred_boxes