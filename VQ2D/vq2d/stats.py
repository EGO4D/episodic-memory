import collections
from re import I
from typing import List, Sequence, Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def barplot(
    x,
    y,
    figsize=(7, 7),
    rotation=None,
    xlabel=None,
    ylabel=None,
    title=None,
    add_grid=False,
    save_path=None,
    **kwargs,
):

    plt.figure(figsize=figsize)
    ax = sns.barplot(x=x, y=y, **kwargs)

    if rotation is not None:
        for item in ax.get_xticklabels():
            item.set_rotation(rotation)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if title is not None:
        plt.title(title)

    if add_grid:
        plt.grid()

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path)
    else:
        plt.show()


def hist(
    x,
    figsize=(7, 7),
    rotation=None,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    title=None,
    save_path=None,
    add_grid=False,
    **kwargs,
):

    plt.figure(figsize=figsize)
    ax = plt.hist(x, **kwargs)

    if rotation is not None:
        for item in ax.get_xticklabels():
            item.set_rotation(rotation)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if title is not None:
        plt.title(title)

    if xlim is not None:
        plt.xlim(*xlim)

    if ylim is not None:
        plt.ylim(*ylim)

    if add_grid:
        plt.grid()

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path)
    else:
        plt.show()


def get_video_coverage(
    video_duration: float,
    clips_start_end: List[Sequence[float]],
    step_size: float = 1.0,
):
    """
    Given a set of clips from a video, estimate how much of the video is
    effectively covered by the clips.

    Args:
        video_duration - Length of the video in seconds
        clips_start_end - A list of (start_sec, end_sec) values for each clip
        step_size - discretization size for measuring coverage
    """
    video_coverage = np.zeros((int(video_duration / step_size) + 1), dtype=np.float32)
    for start_sec, end_sec in clips_start_end:
        start_sec_idx = int(start_sec / step_size)
        end_sec_idx = int(end_sec / step_size) + 1
        video_coverage[start_sec_idx:end_sec_idx] = 1
    video_coverage_sec = video_coverage.sum().item() * step_size
    return video_coverage_sec


def compute_coverage_statistics(
    video_annotations: List[Any], video_uids_to_metadata: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Compute the video hours across all annotations.
    """
    total_coverage = 0.0
    per_split_coverage = collections.defaultdict(float)
    for v in video_annotations:
        video_uid = v["video_uid"]
        video_duration = video_uids_to_metadata[video_uid]["video_duration_sec"]
        clips_start_end = []
        for c in v["clips"]:
            if c["annotation_complete"]:
                clips_start_end.append((c["video_start_sec"], c["video_end_sec"]))
        coverage = get_video_coverage(video_duration, clips_start_end)
        coverage = coverage / 3600.0  # seconds -> hours
        total_coverage += coverage
        per_split_coverage[v["split"]] += coverage

    coverage_stats = {"total_coverage": total_coverage}
    for split, coverage in per_split_coverage.items():
        coverage_stats[f"{split}_coverage"] = coverage

    return coverage_stats


def compute_clip_statistics(video_annotations: List[Any]) -> Dict[str, int]:
    """
    Compute the total number of unique clips annotated.
    """
    all_clips = set()
    per_split_clips = collections.defaultdict(set)
    for v in video_annotations:
        for c in v["clips"]:
            if c["annotation_complete"]:
                all_clips.add(c["clip_uid"])
                per_split_clips[v["split"]].add(c["clip_uid"])

    clip_stats = {"total_clips": len(all_clips)}
    for split, split_clips in per_split_clips.items():
        clip_stats[f"{split}_clips"] = len(split_clips)

    return clip_stats


def compute_query_statistics(video_annotations: List[Any]) -> Dict[str, int]:
    """
    Compute the total number of visual queries annotated.
    """
    total_queries = 0
    per_split_queries = collections.defaultdict(int)
    for v in video_annotations:
        for c in v["clips"]:
            if c["annotation_complete"]:
                for a in c["annotations"]:
                    for q in a["query_sets"].values():
                        if q["is_valid"]:
                            total_queries += 1
                            per_split_queries[v["split"]] += 1

    query_stats = {"total_queries": total_queries}
    for split, split_queries in per_split_queries.items():
        query_stats[f"{split}_queries"] = split_queries
    return query_stats


def compute_scenario_coverage_statistics(
    video_annotations: List[Any],
    video_uids_to_metadata: List[Dict[str, Any]],
    thresh: float = 1.5,
) -> Dict[str, float]:
    """
    Compute the video hours annotated per scenario.
    Note: We only count scenarios for which we have more than `thresh` hours of annotations.
    """
    per_scenario_coverage = collections.defaultdict(float)
    for v in video_annotations:
        video_uid = v["video_uid"]
        video_duration = video_uids_to_metadata[video_uid]["video_duration_sec"]
        clips_start_end = []
        for c in v["clips"]:
            if c["annotation_complete"]:
                clips_start_end.append((c["video_start_sec"], c["video_end_sec"]))
        coverage = get_video_coverage(video_duration, clips_start_end)
        coverage = coverage / 3600.0  # seconds -> hours
        for scenario in video_uids_to_metadata[video_uid]["scenarios"]:
            per_scenario_coverage[scenario] += coverage
    per_scenario_coverage = {
        k: v for k, v in per_scenario_coverage.items() if v > thresh
    }

    return per_scenario_coverage


def compute_university_coverage_statistics(
    video_annotations: List[Any],
    video_uids_to_metadata: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute the video hours annotated per university.
    """
    per_university_coverage = collections.defaultdict(float)
    for v in video_annotations:
        video_uid = v["video_uid"]
        video_duration = video_uids_to_metadata[video_uid]["video_duration_sec"]
        clips_start_end = []
        for c in v["clips"]:
            if c["annotation_complete"]:
                clips_start_end.append((c["video_start_sec"], c["video_end_sec"]))
        coverage = get_video_coverage(video_duration, clips_start_end)
        coverage = coverage / 3600.0  # seconds -> hours
        university = video_uids_to_metadata[video_uid]["source"]
        per_university_coverage[university] += coverage

    return per_university_coverage


def compute_query_to_response_separation_statistics(
    video_annotations: List[Any],
) -> List[int]:
    """
    Compute the set of query to response separation in # frames.
    """
    q2r_separation = []
    for v in video_annotations:
        for c in v["clips"]:
            if c["annotation_complete"]:
                for a in c["annotations"]:
                    for q in a["query_sets"].values():
                        if q["is_valid"]:
                            qfno = q["query_frame"]
                            rtfno = max(
                                [rf["frame_number"] for rf in q["response_track"]]
                            )
                            assert qfno - rtfno > 0
                            q2r_separation.append(qfno - rtfno)

    return q2r_separation


def compute_response_track_length_statistics(
    video_annotations: List[Any],
) -> List[int]:
    """
    Compute the set of response track lengths in # frames.
    """
    rt_lengths = []
    for v in video_annotations:
        for c in v["clips"]:
            if c["annotation_complete"]:
                for a in c["annotations"]:
                    for q in a["query_sets"].values():
                        if q["is_valid"]:
                            rt_max = max(
                                [rf["frame_number"] for rf in q["response_track"]]
                            )
                            rt_min = min(
                                [rf["frame_number"] for rf in q["response_track"]]
                            )
                            rt_lengths.append(rt_max - rt_min + 1)
    return rt_lengths


def compute_response_track_location_statistics(
    video_annotations: List[Any],
) -> List[Sequence[float]]:
    """
    Compute the set of response track lengths in # frames.
    """
    rt_bbox_locations = []
    for v in video_annotations:
        for c in v["clips"]:
            if c["annotation_complete"]:
                for a in c["annotations"]:
                    for q in a["query_sets"].values():
                        if q["is_valid"]:
                            for rf in q["response_track"]:
                                oh = rf["original_height"]
                                ow = rf["original_width"]
                                xs = rf["x"] / ow
                                ys = rf["y"] / oh
                                xe = (rf["x"] + rf["width"]) / ow
                                ye = (rf["y"] + rf["height"]) / oh
                                rt_bbox_locations.append((xs, ys, xe, ye))
    return rt_bbox_locations
