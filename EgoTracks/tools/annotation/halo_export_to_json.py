"""
Train_net for Tracking

Run this command to test locally:
>>> buck run @mode/dev-nosan //vision/fair_accel/pixar_env/pixar_environment/tracking/tools:halo_export_to_json -- --print-passing-details
"""

import json
import logging
from typing import NamedTuple, Set

import numpy as np
from analytics.bamboo import Bamboo as bb
from iopath.common.file_io import PathManager
from iopath.fb.manifold import ManifoldPathHandler
from tqdm import tqdm
from tracking.tools.annotation.annotation_utils import (
    filter_payload_attribute,
    find_occurance_by_frame_number,
    merge_vq_annotations,
    seperate_occurrances,
)


class HaloExport2JsonParams(NamedTuple):
    project_id: int = 1311813402564819
    task_ids: Set = (
        3236903243296518,
        1553405651708980,
        312475530965750,
        2844428002517867,
        968454790525495,
        425734286162311,
        548453387055109,
        1448847242191253,
    )
    output_path: str = (
        "manifold://tracking/tree/data/lt_tracking_annotation/lt_tracking_val.json"
    )
    ego4d_annotation_path: str = (
        "manifold://tracking/tree/ego4d/v1/annotations/vq_val.json"
    )
    halo_export_table: str = "xdb.tracking.halo_export"


def main():
    pathmgr = PathManager()
    pathmgr.register_handler(ManifoldPathHandler())
    params = HaloExport2JsonParams()

    query_halo_export = f"""
    WITH m AS (
        SELECT
            *
        FROM xdb.halo.halo_metric
        WHERE
            task_id IN {params.task_ids}
    )

    SELECT
        h.external_object_id,
        h.annotator_id,
        e.clip_uid,
        e.label,
        h.payload,
        h.job_id,
        COALESCE(m.numerical_value, 0) AS iou
    FROM xdb.tracking.ego4d_vq_response_track AS e
    LEFT JOIN {params.halo_export_table} AS h
        ON e.id = h.external_object_id
    LEFT JOIN xdb.halo.halo_metric AS m
        ON m.response_id = h.response_id
    WHERE
        h.task_id IN {params.task_ids}
        AND h.project_id = {params.project_id}
        AND h.is_rejected = 0
    """

    # Read annotation from halo_export
    logging.error(query_halo_export)
    res = bb.query_presto("aml", query_halo_export)
    res.sort_values("external_object_id")
    tracklet_ids = np.unique(res["external_object_id"])
    tracklet_ids = set(tracklet_ids)
    logging.error(f"Total jobs: {len(tracklet_ids)}")

    # Read original VQ annotation from Ego4D
    annotation_path = params.ego4d_annotation_path
    with pathmgr.open(annotation_path, "r") as f:
        vq_annotations = json.load(f)
        vq_annotations = merge_vq_annotations(vq_annotations)

    merged_vq_annotations = vq_annotations.copy()
    vq_video_dict = {
        x["video_uid"]: x["clips"] for x in merged_vq_annotations["videos"]
    }
    vq_clip2video = {
        clip["source_clip_uid"]: x["video_uid"]
        for x in merged_vq_annotations["videos"]
        for clip in x["clips"]
    }

    # Export lt tracking annotation into the original VQ, as an extra field "lt_track"
    for tracklet_id in tqdm(tracklet_ids, total=len(tracklet_ids)):
        # tracklet_id = tracklet_ids[0]
        mr_res = res[res["external_object_id"] == tracklet_id]
        gt = mr_res[mr_res["iou"] == mr_res["iou"].max()].iloc[0]
        payload = gt["payload"]
        payload = json.loads(payload)["payload"]
        payload = filter_payload_attribute(payload)

        target_id = gt["label"].split("_")[0]
        clip_uid = gt["clip_uid"]

        video_uid = vq_clip2video[clip_uid]
        clips = vq_video_dict[video_uid]
        clip = [c for c in clips if c["source_clip_uid"] == clip_uid][0]
        query_sets = clip["annotations"][0]["query_sets"]
        query_sets[target_id]["lt_track"] = payload

        # Also export a clip for visual clip
        payload_dict = {p["frame_number"]: p for p in payload}
        frame_numbers = payload_dict.keys()
        vq_frame_number = query_sets[target_id]["visual_crop"]["frame_number"]
        occurances = seperate_occurrances(frame_numbers)

        visual_clip_frame_numbers = find_occurance_by_frame_number(
            occurances, vq_frame_number
        )
        if visual_clip_frame_numbers is None:
            logging.error(
                f"Warning: {clip_uid}, {gt['label']} does not have any overlap with visual_Crop"
            )
            visual_clip_payload = [query_sets[target_id]["visual_crop"]]
        else:
            visual_clip_payload = [payload_dict[fn] for fn in visual_clip_frame_numbers]
        query_sets[target_id]["visual_clip"] = visual_clip_payload

    logging.error(f"Write output to {params.output_path,}")
    with pathmgr.open(
        params.output_path,
        "w",
    ) as outfile:
        json.dump(merged_vq_annotations, outfile)


if __name__ == "__main__":
    main()
