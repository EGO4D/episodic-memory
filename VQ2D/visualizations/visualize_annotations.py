import argparse
import json
import os

import cv2
import imageio
import numpy as np
import pims


def _get_box(annot_box):
    x, y, w, h = annot_box["x"], annot_box["y"], annot_box["width"], annot_box["height"]
    return (int(x), int(y), int(x + w), int(y + h))


def extract_crop_from_image(image, box):
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    # Add padding to make it square
    pad_l, pad_r, pad_t, pad_b = 0, 0, 0, 0
    if h > w:
        pad_l = (h - w) // 2
        pad_r = (h - w) - pad_l
    if w > h:
        pad_t = (w - h) // 2
        pad_b = (w - h) - pad_t
    crop = np.pad(
        crop,
        ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
        mode="constant",
        constant_values=255,
    )
    return crop


def draw_box_on_image(image, box, color=(255, 0, 0), thickness=5):
    x1, y1, x2, y2 = [int(b) for b in box]
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=thickness)
    return image


def draw_border(image, color=(255, 0, 0), thickness=5):
    x1, y1 = 0, 0
    x2, y2 = image.shape[1] - 1, image.shape[0] - 1
    return draw_box_on_image(image, [x1, y1, x2, y2], color=color, thickness=thickness)


def scale_im_height(image, H):
    im_H, im_W = image.shape[:2]
    W = int(1.0 * H * im_W / im_H)
    return cv2.resize(image, (W, H))


def visualize_query_set(video_reader, qset, save_height=640):
    qf_fno = qset["query_frame"]
    vc_fno = qset["visual_crop"]["frame_number"]
    last_fno = max(qf_fno, vc_fno)
    vc_box = _get_box(qset["visual_crop"])
    rt_fnos = [rf["frame_number"] for rf in qset["response_track"]]
    rt_boxes = [_get_box(rf) for rf in qset["response_track"]]
    oW = qset["visual_crop"]["original_width"]
    oH = qset["visual_crop"]["original_height"]
    # Visualize visual crop
    vc_frame = np.copy(video_reader[vc_fno])
    ## Scale up to original resolution
    vc_frame = scale_im_height(vc_frame, oH)
    vc_im = extract_crop_from_image(vc_frame, vc_box)
    ## Add text header
    tx_height = 50
    ob_title = qset["object_title"]
    tx_im = get_text_box(ob_title, (tx_height, vc_im.shape[1]))
    vc_im = np.concatenate([tx_im, vc_im], axis=0)
    # Visualize frames in the response track
    rt_ims = []
    for rf_fno, rf_box in zip(rt_fnos, rt_boxes):
        rf_frame = np.copy(video_reader[rf_fno])
        ## Scale up to original resolution
        rf_frame = scale_im_height(rf_frame, oH)
        rf_im = draw_box_on_image(rf_frame, rf_box, color=(0, 255, 0), thickness=8)
        rf_im = draw_border(rf_im, color=(0, 255, 0), thickness=15)
        ## Scale down to save height
        rf_im = scale_im_height(rf_im, save_height)
        rt_ims.append(rf_im)
    # Visualize frames after the response track till the query frame
    post_rt_ims = []
    for i in range(rt_fnos[-1], qf_fno):
        # Concatenate vc_plot_im to the right
        frame = scale_im_height(video_reader[i], save_height)
        post_rt_ims.append(frame)
    rt_ims = rt_ims + post_rt_ims
    return rt_ims, vc_im


def get_mp4_writer(path, fps, output_params=["-crf", "31"]):
    writer = imageio.get_writer(
        path,
        codec="h264",
        fps=fps,
        quality=None,
        pixelformat="yuv420p",
        bitrate=0,  # Setting bitrate to 0 is required to activate -crf
        output_params=output_params,
    )
    return writer


def save_video(frames, path, fps):
    writer = get_mp4_writer(path, fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def get_text_box(text, shape, fg_color=(255, 255, 255), bg_color=(0, 0, 0)):
    text_im = np.zeros((*shape, 3), dtype=np.uint8)
    text_im[:, :] = bg_color
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    line_thickness = 2
    text_im = cv2.putText(
        text_im,
        text,
        (2, shape[0] - 10),
        font,
        font_scale,
        fg_color,
        line_thickness,
        cv2.LINE_AA,
    )
    return text_im


def visualize_annotation(clip_path, rt_save_path, crop_save_path, qset):
    """
    Visualizes an annotation from the visual-queries task
    """
    video_reader = pims.Video(clip_path)
    # Visualize annotations for 3 query sets
    rt_frames, vc_im = visualize_query_set(video_reader, qset)
    save_video(rt_frames, rt_save_path, 5.0)
    imageio.imwrite(crop_save_path, vc_im)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annot-path", type=str, required=True)
    parser.add_argument("--clips-root", type=str, required=True)
    parser.add_argument("--vis-save-root", type=str, default="./videos")

    args = parser.parse_args()

    with open(args.annot_path, "r") as fp:
        annotations = json.load(fp)

    os.makedirs(args.vis_save_root, exist_ok=True)

    for v in annotations["videos"]:
        for c in v["clips"]:
            cuid = c["clip_uid"]
            for a_idx, a in enumerate(c["annotations"]):
                for qset_id, qset in a["query_sets"].items():
                    if not qset["is_valid"]:
                        continue
                    qf_fno = qset["query_frame"]
                    rt_last_fno = max(
                        [rf["frame_number"] for rf in qset["response_track"]]
                    )
                    rtsp = f"{args.vis_save_root}/{cuid}_{a_idx:05d}_{qset_id}_rt.mp4"
                    csp = f"{args.vis_save_root}/{cuid}_{a_idx:05d}_{qset_id}_crop.png"
                    clip_path = f"{args.clips_root}/{cuid}.mp4"
                    if not os.path.isfile(clip_path):
                        print(f"======> Clip {clip_path} is missing...")
                        continue
                    visualize_annotation(clip_path, rtsp, csp, qset)
