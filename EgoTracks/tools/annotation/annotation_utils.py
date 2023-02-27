from typing import Dict, List

import numpy as np


def filter_payload_attribute(payload: List):
    res = []
    for item in payload:
        interpolated = item.get("interpolated", False)

        if not interpolated:
            res.append(
                {
                    "frame_number": item["frameNumber"],
                    "x": item["x"],
                    "y": item["y"],
                    "width": item["width"],
                    "height": item["height"],
                }
            )

    return res


def payload2frame_bbox_dict(payload: List):
    frame_bbox_dict = {}
    for item in payload:
        interpolated = item.get("interpolated", False)
        frame_number = item["frameNumber"]
        x, y, w, h = item["x"], item["y"], item["width"], item["height"]

        if not interpolated:
            frame_bbox_dict[frame_number] = [x, y, w, h]
    return frame_bbox_dict


def find_occurance_by_frame_number(occurances: List, frame_number: int):
    for occ in occurances:
        start = occ[0]
        end = occ[-1]
        if start <= frame_number <= end:
            return occ

    return None


def seperate_occurrances(frame_numbers: np.ndarray, min_frame_per_occurance=1):
    res = []
    cur = []
    prev = None

    for frame_number in sorted(frame_numbers):
        if prev is None:
            prev = frame_number
            cur.append(prev)
            continue

        if frame_number - prev != 1:
            if len(cur) >= min_frame_per_occurance:
                res.append(cur)
            cur = []

        prev = frame_number
        cur.append(frame_number)

    # Need to make sure the last occurance is appended
    if len(cur) >= min_frame_per_occurance:
        res.append(cur)

    return res


def merge_vq_annotations(vq_annotations: Dict):
    """
    Because the vq annotation contains multiple "annotations" for each clip,
    and each "annotation" contains three query set objects indexed from 1 to 3.
    We need to merge them all into one "annotation".
    """
    vq_annotations = vq_annotations.copy()
    for video in vq_annotations["videos"]:
        for clip in video["clips"]:
            cnt = 0
            base_query_sets = clip["annotations"][0]["query_sets"].copy()

            for _, anno in enumerate(clip["annotations"]):
                for target_id, query_set in anno["query_sets"].items():
                    base_query_sets[str(cnt + int(target_id))] = query_set

                # We would like to keep original target_id for each query set,
                # and the target_ids among different "annotation" can overlap.
                # So, we only need to make sure the start point for different
                # "annotation"s are different.
                cnt += len(anno["query_sets"])

            clip["annotations"] = [{"query_sets": base_query_sets}]

    return vq_annotations
