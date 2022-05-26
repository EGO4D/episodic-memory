import argparse
import json
import tqdm


def validate_model_predictions(model_predictions, test_annotations):
    assert type(model_predictions) == type({})
    for key in ["version", "challenge", "results"]:
        assert key in model_predictions.keys()
    assert model_predictions["version"] == "1.0"
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
                    k: v
                    for k, v in clip_annot["query_sets"].items()
                    if v["is_valid"]
                }
                valid_query_set_preds = {
                    k: v
                    for k, v in clip_pred["query_sets"].items()
                    if clip_annot["query_sets"][k]["is_valid"]
                }
                assert(
                    set(list(valid_query_set_preds.keys())) == \
                    set(list(valid_query_set_annots.keys()))
                )
                for qset_id, qset in clip_pred["query_sets"].items():
                    assert type(qset) == type({})
                    for key in ["bboxes", "score"]:
                        assert key in qset
                    pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-unannotated-path", type=str, required=True)
    parser.add_argument("--test-predictions-path", type=str, required=True)

    args = parser.parse_args()

    with open(args.test_unannotated_path, "r") as fp:
        test_annotations = json.load(fp)
    with open(args.test_predictions, "r") as fp:
        model_predictions = json.load(fp)
    validate_model_predictions(model_predictions, test_annotations)