import argparse
import glob
import gzip
import json
from collections import defaultdict

import tqdm
from vq2d.metrics import compute_visual_query_metrics
from vq2d.structures import (
    BBox,
    ResponseTrack,
)


NUM_PARTS = 100


def merge_results(args):
    stats_paths = sorted(glob.glob(f"{args.stats_dir}/*.json.gz"))
    assert len(stats_paths) == NUM_PARTS
    results = defaultdict(list)
    for path in tqdm.tqdm(stats_paths):
        with gzip.open(path, "rt") as fp:
            data = json.load(fp)
        predictions = data["predictions"]
        results["predicted_response_track"] += predictions["predicted_response_track"][
            0
        ]
        results["ground_truth_response_track"] += predictions[
            "ground_truth_response_track"
        ]
        results["visual_crop"] += predictions["visual_crop"]
        results["dataset_uids"] += predictions["dataset_uids"]
        results["accessed_frames"] += predictions["accessed_frames"]
        results["total_frames"] += predictions["total_frames"]
    ################################################################################################
    # Save results
    with gzip.open(f"{args.stats_dir}/merged_output.json.gz", "wt") as fp:
        json.dump(results, fp)
    ################################################################################################
    predicted_response_track = [
        [ResponseTrack.from_json(rt) for rt in rts]
        for rts in results["predicted_response_track"]
    ]
    ground_truth_response_track = [
        ResponseTrack.from_json(rt) for rt in results["ground_truth_response_track"]
    ]
    visual_crop_boxes = [BBox.from_json(vc) for vc in results["visual_crop"]]
    accessed_frames = results["accessed_frames"]
    total_frames = results["total_frames"]
    metrics = compute_visual_query_metrics(
        predicted_response_track,
        ground_truth_response_track,
        visual_crop_boxes,
        accessed_frames,
        total_frames,
    )
    print("==========> Retrieval performance")
    for k, v in metrics.items():
        print(f"{k:<40s} | {v:8.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-dir", type=str, required=True)

    args = parser.parse_args()

    merge_results(args)
