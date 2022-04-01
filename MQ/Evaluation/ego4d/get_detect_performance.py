
import os
from .eval_detection import ANETdetection

def run_evaluation(ground_truth_filename, prediction_filename,
         subset='test', tiou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
         verbose=True):

    anet_detection = ANETdetection(ground_truth_filename, prediction_filename,
                                   subset=subset, tiou_thresholds=tiou_thresholds,
                                   verbose=verbose, check_status=False)
    mAPs, average_mAP = anet_detection.evaluate()

    for (tiou, mAP) in zip(tiou_thresholds, mAPs):
        print("mAP at tIoU {} is {}".format(tiou, mAP))



def evaluation_detection(opt):

    run_evaluation(ground_truth_filename = opt["clip_anno"],
                   prediction_filename = os.path.join(opt["output_path"], opt["detect_result_file"]),
                   subset=opt['infer_datasplit'], tiou_thresholds=opt['tIoU_thr'])