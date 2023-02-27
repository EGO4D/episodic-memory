import argparse
import json

import tqdm
from vq2d.metrics.metrics import compute_visual_query_metrics
from vq2d.structures import ResponseTrack, BBox


def validate_model_predictions(model_predictions, test_annotations):
    assert type(model_predictions) == type({})
    for key in ["version", "challenge", "results"]:
        assert key in model_predictions.keys()
    assert model_predictions["version"] == test_annotations["version"]
    assert model_predictions["challenge"] == "ego4d_vq2d_challenge"
    assert type(model_predictions["results"]) == type({})
    assert "videos" in model_predictions["results"]

    video_annotations = test_annotations["videos"]
    video_predictions = model_predictions["results"]["videos"]

    assert len(video_predictions) == len(video_annotations)

    n_samples = 0
    for v in video_annotations:
        for c in v["clips"]:
            for a in c["annotations"]:
                for _, q in a["query_sets"].items():
                    if q["is_valid"]:
                        n_samples += 1

    pbar = tqdm.tqdm(total=n_samples, desc="Validating user predictions")
    for vannot, vpred in zip(video_annotations, video_predictions):
        assert type(vpred) == type({})
        for key in ["video_uid", "clips"]:
            assert key in vpred
        assert vannot["video_uid"] == vpred["video_uid"]
        assert type(vpred["clips"]) == type([])
        assert len(vannot["clips"]) == len(vpred["clips"])
        for cpreds in vpred["clips"]:
            assert type(cpreds) == type({})
            for key in ["clip_uid", "predictions"]:
                assert key in cpreds
        clips_annots = vannot["clips"]
        clips_preds = vpred["clips"]
        for clip_annots, clip_preds in zip(clips_annots, clips_preds):
            assert clip_annots["clip_uid"] == clip_preds["clip_uid"]
            assert type(clip_preds["predictions"]) == type([])
            assert len(clip_preds["predictions"]) == len(clip_annots["annotations"])
            for clip_annot, clip_pred in zip(
                clip_annots["annotations"], clip_preds["predictions"]
            ):
                assert type(clip_pred) == type({})
                assert "query_sets" in clip_pred
                valid_query_set_annots = {
                    k: v for k, v in clip_annot["query_sets"].items() if v["is_valid"]
                }
                valid_query_set_preds = {
                    k: v
                    for k, v in clip_pred["query_sets"].items()
                    if clip_annot["query_sets"][k]["is_valid"]
                }
                assert set(list(valid_query_set_preds.keys())) == set(
                    list(valid_query_set_annots.keys())
                )
                for qset_id, qset in clip_pred["query_sets"].items():
                    assert type(qset) == type({})
                    for key in ["bboxes", "score"]:
                        assert key in qset
                    pbar.update()


def evaluate(gt_file, pred_file):
    print("Starting Evaluation.....")

    with open(gt_file, "r") as fp:
        gt_annotations = json.load(fp)
    with open(pred_file, "r") as fp:
        model_predictions = json.load(fp)

    # Validate model predictions
    validate_model_predictions(model_predictions, gt_annotations)

    # Convert test annotations, model predictions to the correct format
    predicted_response_tracks = []
    annotated_response_tracks = []
    visual_crop_boxes = []
    for vanno, vpred in zip(
        gt_annotations["videos"], model_predictions["results"]["videos"]
    ):
        for clip_annos, clip_preds in zip(vanno["clips"], vpred["clips"]):
            for clip_anno, clip_pred in zip(
                clip_annos["annotations"], clip_preds["predictions"]
            ):
                qset_ids = list(clip_anno["query_sets"].keys())
                for qset_id in qset_ids:
                    if not clip_anno["query_sets"][qset_id]["is_valid"]:
                        continue
                    q_anno = clip_anno["query_sets"][qset_id]
                    q_pred = clip_pred["query_sets"][qset_id]
                    rt_pred = ResponseTrack.from_json(q_pred)
                    rt_anno = []
                    for rf in q_anno["response_track"]:
                        rt_anno.append(
                            BBox(
                                rf["frame_number"],
                                rf["x"],
                                rf["y"],
                                rf["x"] + rf["width"],
                                rf["y"] + rf["height"],
                            )
                        )
                    rt_anno = ResponseTrack(rt_anno)
                    vc = q_anno["visual_crop"]
                    vc_bbox = BBox(
                        vc["frame_number"],
                        vc["x"],
                        vc["y"],
                        vc["x"] + vc["width"],
                        vc["y"] + vc["height"],
                    )
                    predicted_response_tracks.append([rt_pred])
                    annotated_response_tracks.append(rt_anno)
                    visual_crop_boxes.append(vc_bbox)

    # Perform evaluation
    pair_metrics = compute_visual_query_metrics(
        predicted_response_tracks,
        annotated_response_tracks,
        visual_crop_boxes,
    )

    print("Evaluating VQ2D performance")
    for pair_name, metrics in pair_metrics.items():
        print("-" * 20)
        print(pair_name)
        print("-" * 20)
        metrics = {
            "tAP": metrics["Temporal AP                    @ IoU=0.25:0.95"],
            "tAP @ IoU=0.25": metrics["Temporal AP                    @ IoU=0.25     "],
            "stAP": metrics["SpatioTemporal AP              @ IoU=0.25:0.95"],
            "stAP @ IoU=0.25": metrics[
                "SpatioTemporal AP              @ IoU=0.25     "
            ],
            "success": metrics["Success (max scr)              @ IoU=0.05     "],
            "recovery %": metrics["Tracking % recovery (max scr)  @ IoU=0.50     "],
        }
        for k, v in metrics.items():
            print(f"{k:<20s} | {v:>10.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", required=True, type=str)
    parser.add_argument("--pred-file", required=True, type=str)

    args = parser.parse_args()

    evaluate(args.gt_file, args.pred_file)
