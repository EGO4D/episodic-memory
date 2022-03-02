import sys
sys.dont_write_bytecode = True
import torch
import torch.nn.parallel
from Utils.dataset import VideoDataSet
from Models.VSGN import VSGN
import pandas as pd
from joblib import Parallel, delayed
import sys
sys.dont_write_bytecode = True
import torch.nn.parallel
import os
import Utils.opts as opts

import datetime
import numpy as np

torch.manual_seed(21)


# Infer all data
def Infer_SegTAD(opt):
    model = VSGN(opt)
    model = torch.nn.DataParallel(model).cuda()
    if not os.path.exists(opt["checkpoint_path"] + "/best.pth.tar"):
        print("There is no checkpoint. Please train first!!!")
    else:
        checkpoint = torch.load(opt["checkpoint_path"] + "/best.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    proposal_path = os.path.join(opt["output_path"], opt["prop_path"])

    if not os.path.exists(proposal_path):
        os.mkdir(proposal_path)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset=opt['infer_datasplit'], mode="inference"),
                                              batch_size=opt["batch_size"], shuffle=False,
                                              num_workers=16, pin_memory=True, drop_last=False)

    with torch.no_grad():
        for i, (index_list, input_data, num_frms) in enumerate(test_loader):

            infer_batch_selectprop(model, index_list, input_data, test_loader, proposal_path, num_frms)


# Infer one batch of data, for the model with proposal selection
def infer_batch_selectprop(model,
                           index_list,
                           input_data,
                           test_loader,
                           proposal_path,
                           num_frms):

    loc_dec, score_dec, loc_adjusted, pred_action, pred_start, pred_end = model(input_data.cuda(), num_frms)

    # Move variables to output to CPU
    loc_adjusted_batch = loc_adjusted.detach().cpu().numpy()
    score_dec_batch = score_dec.detach().cpu().numpy()
    pred_action_batch = pred_action.detach().cpu().numpy()
    pred_start_batch = pred_start.detach().cpu().numpy()
    pred_end_batch = pred_end.detach().cpu().numpy()
    num_frms_batch = num_frms.detach().cpu().numpy()


    Parallel(n_jobs=len(index_list))(
        delayed(infer_v_asis)(
            opt,
            video=test_loader.dataset.clip_dict[test_loader.dataset.clip_list[full_idx]],
            score_dec_v = score_dec_batch[batch_idx],
            loc_dec_v = loc_adjusted_batch[batch_idx],
            pred_action_v = pred_action_batch[batch_idx],
            pred_start_v = pred_start_batch[batch_idx],
            pred_end_v = pred_end_batch[batch_idx],
            proposal_path = proposal_path,
            num_frms_v = num_frms_batch[batch_idx]

        ) for batch_idx, full_idx in enumerate(index_list))



def infer_v_asis(*args, **kwargs):
    tscale = args[0]["temporal_scale"]
    loc_pred_v = kwargs['loc_dec_v']
    score_dec_v = kwargs['score_dec_v']
    pred_start_v = kwargs['pred_start_v']
    pred_end_v = kwargs['pred_end_v']
    proposal_path = kwargs['proposal_path']
    num_frms_v = kwargs['num_frms_v']
    thresh = 0.000000005

    clip_name = kwargs['video']['clip_id']
    fps = kwargs['video']['fps']

    loc_pred_v[:, 0] = loc_pred_v[:, 0].clip(min=0, max=tscale - 1)
    loc_pred_v[:, 1] = loc_pred_v[:, 1].clip(min=0, max=tscale - 1)

    start_score = (pred_start_v[np.ceil(loc_pred_v[:, 0]).astype('int32')] + pred_start_v[
        np.floor(loc_pred_v[:, 0]).astype('int32')]) / 2
    end_score = (pred_end_v[np.ceil(loc_pred_v[:, 1]).astype('int32')] + pred_end_v[
        np.floor(loc_pred_v[:, 1]).astype('int32')]) / 2

    score_stage2 = start_score * end_score

    loc_pred_v[:, 0] = loc_pred_v[:, 0].clip(min=0, max=num_frms_v - 1)
    loc_pred_v[:, 1] = loc_pred_v[:, 1].clip(min=0, max=num_frms_v - 1)

    new_props = []
    for j in range(1, opt['decoder_num_classes']):
        inds = (score_dec_v[:, j] > thresh)
        scores = (score_dec_v[:, j] * score_stage2)[inds]
        locations = loc_pred_v[inds, :]
        labels = np.array([j] * locations.shape[0])
        cls_dets = np.concatenate((locations, scores[:, None], labels[:, None]), axis=1)
        keep = nms(cls_dets, opt['nms_thr'])
        if (len(keep) > 0):
            cls_dets = cls_dets[keep]

        new_props.append(cls_dets)

    new_props = np.concatenate(new_props, axis=0)
    new_props[:, :2] = (new_props[:, :2]) / fps

    col_name = ["xmin", "xmax", "score", "label"]
    new_df = pd.DataFrame(new_props, columns=col_name)
    path = proposal_path + "/" + clip_name + ".csv"
    new_df.to_csv(path, index=False)


def nms(dets, thresh=0.4):
    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    scores = dets[:, 2]
    lengths = x2 - x1
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


if __name__ == '__main__':

    opt = opts.parse_opt()
    opt = vars(opt)

    print(opt)

    if not os.path.exists(opt["output_path"]):
        os.makedirs(opt["output_path"])

    print(datetime.datetime.now())
    print("---------------------------------------------------------------------------------------------")
    print("1. Inference starts!")
    print("---------------------------------------------------------------------------------------------")

    Infer_SegTAD(opt)

    print("Inference finishes! \n")

