
import torch
from .BoxCoder import BoxCoder
import torch.nn.functional as F

class Pred_loc_cls(object):
    def __init__(self, opt):
        super(Pred_loc_cls, self).__init__()

        self.pre_nms_thresh = 0.00
        self.pre_nms_top_n = 10000
        self.num_classes = opt['decoder_num_classes']

        self.box_coder = BoxCoder(opt)

    def __call__(self, cls_pred_dec, reg_pred_dec, anchors):
        bs = cls_pred_dec[0].shape[0]
        anchors = [anchor.unsqueeze(0).repeat(bs, 1, 1).to(device=cls_pred_dec[0].device) for anchor in anchors]
        all_anchors = torch.cat(anchors, dim=1)          # bs, levels*positions*scales, left-right

        cls_pred_dec = [cls_pred_dec[i] for i in range(len(cls_pred_dec)-1, -1, -1)]
        reg_pred_dec = [reg_pred_dec[i] for i in range(len(reg_pred_dec)-1, -1, -1)]

        loc_dec, _ = self._call_one_stage(cls_pred_dec, reg_pred_dec, all_anchors)
        _, score_dec = self._call_one_stage(cls_pred_dec, reg_pred_dec, torch.stack(loc_dec, dim=0))


        return torch.stack(score_dec, dim=0), torch.stack(loc_dec, dim=0)

    def _call_one_stage(self, cls_pred, reg_pred, all_anchors):

        N = cls_pred[0].shape[0]

        cls_pred = F.softmax(torch.cat(cls_pred, dim=2).permute(0, 2, 1).reshape(N, -1, self.num_classes), dim=-1)  # bs, levels*positions*scales, num_cls
        reg_pred = torch.cat(reg_pred, dim=2).permute(0, 2, 1).reshape(N, -1, 2)   # bs, levels*positions*scales, 2


        candidate_inds = cls_pred > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        loc_res = []
        score_res = []
        for cls_seq, reg_seq, anchor_seq, pre_nms_top_n_seq, candidate_inds_seq in zip(cls_pred, reg_pred, all_anchors, pre_nms_top_n, candidate_inds):

            loc_pred = self.box_coder.decode(
                reg_seq.view(-1, 2),        # levels*positions*scales, 2
                anchor_seq.view(-1, 2)      # levels*positions*scales, 2
            )

            score_pred = cls_seq

            loc_res.append(loc_pred)
            score_res.append(score_pred)

        return loc_res, score_res


    def cat_boxlist(self, bboxes):
        """
        Concatenates a list of BoxList (having the same image size) into a
        single BoxList

        Arguments:
            bboxes (list[BoxList])
        """
        assert isinstance(bboxes, (list, tuple))
        assert all(isinstance(bbox, dict) for bbox in bboxes)

        res = {}
        res['loc'] = torch.ones([0, 2])
        for bb in bboxes:
            res['loc'] = torch.cat(res['loc'], bb['loc'], dim=0)

        return res
