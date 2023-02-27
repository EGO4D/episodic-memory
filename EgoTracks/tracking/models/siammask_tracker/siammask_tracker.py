import logging
import math
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tracking.models.siammask_tracker.custom import Custom
from tracking.models.siammask_tracker.utils import generate_anchor, get_subwindow
from tracking.models.single_object_tracker import SingleObjectTracker
from tracking.utils.bbox_helper import cxywh_2_xywh, xywh_2_cxywh
from tracking.utils.env import pathmgr
from tracking.utils.load_helper import load_pretrain
from tracking.utils.types import Params


class SiamMaskTracker(SingleObjectTracker):
    def __init__(
        self,
        cfg: Params,
        device: torch.device = None,
    ):
        # Define siammask model
        # TODO: Use config file to store hyperparamenters, including pretrained model
        logging.info(cfg)
        anchor_cfg = cfg.SiamMask.test.anchor_cfg
        pretrained_path = cfg.model_path
        self.anchor_num = len(anchor_cfg["ratios"]) * len(anchor_cfg["scales"])
        self.template_size: int = cfg.SiamMask.test.template_size
        self.search_size: int = cfg.SiamMask.test.search_size
        self.confidence_thresh: float = cfg.SiamMask.test.confidence_thresh
        self.out_size: int = cfg.SiamMask.test.out_size
        self.base_size: int = cfg.SiamMask.test.base_size
        self.total_stride: int = cfg.SiamMask.test.total_stride
        self.seg_thr: float = cfg.SiamMask.test.seg_thr
        self.context_amount: float = cfg.SiamMask.test.context_amount
        # This is different from original work, original 0.09
        self.penalty_k: float = cfg.SiamMask.test.penalty_k
        self.windowing: str = cfg.SiamMask.test.windowing
        # This is different from original work, original 0.39
        self.window_influence: float = cfg.SiamMask.test.window_influence
        # This is different from original work, original 0.38
        self.lr: float = cfg.SiamMask.test.lr
        # If the input image needs to be cropped to smaller patches,
        # this variable controls how many these patches we can send at a time.
        self.test_batch_size: int = cfg.SiamMask.test.test_batch_size

        # TODO: Move to GPU/CPU outside of model
        self.device = device if device is not None else torch.device("cpu")
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        siammask = Custom(anchors=anchor_cfg)

        if pretrained_path is not None:
            logging.info(f"Loading pretrained model from {pretrained_path} ...")
            with pathmgr.open(pretrained_path, "rb") as f:
                siammask = load_pretrain(siammask, f)
        super(SiamMaskTracker, self).__init__(siammask)

        self.model = self.model.to(self.device)

        # Whether predict mask
        # This was put at the end to override parent function
        self.enable_mask: bool = cfg.SiamMask.test.enable_mask
        self.enable_refine: bool = cfg.SiamMask.test.enable_refine
        self.multiscale_list = [1]
        self.is_search_local: bool = cfg.is_search_local

        # When doing local search as the original SiamMask,
        # multiscale is not supported!
        if self.is_search_local and self.multiscale_list != [1]:
            raise NotImplementedError

    def init_tracker(self, img: torch.Tensor, meta_data: Dict):
        """
        This function is used to initilize SOT with first frame annotation

        Args:
            image: this is one image where we used to initilize the tracker, [1, 3, H, W]
            meta: a dictionary contains information of target objects in this image.
                It should contain "target_bbox": [x, y, w, h], "target_id": str
        Returns:
            None
        """
        target_bbox = meta_data["target_bbox"]
        target_id = meta_data["target_id"]

        img_npy = img.permute(0, 2, 3, 1).contiguous().numpy()[0]
        self.instance_templates.add(
            target_id, img_npy, bbox=target_bbox, features=None, cropped=False
        )
        _, _, img_h, img_w = img.shape

        # Query all instances if self.query_list is None.
        instance_lists = self.instance_templates.get_instance_ids()

        # Loop for all query instances.
        for instance_id in instance_lists:
            try:
                templates = self.instance_templates.get_instance(instance_id)
            except Exception:
                raise ValueError(
                    "Instance_id {} has not registered yet!".format(instance_id)
                )

            # Loop for all templates of the instance.
            for idx, template in enumerate(templates):
                # Only update template features if they have not been extracted.
                if template.features is not None:
                    continue

                if template.cropped:
                    raise NotImplementedError
                else:
                    # Convert bbox to center position and add context area.
                    x, y, w, h = xywh_2_cxywh(template.bbox)
                    pos = np.array([x, y])
                    size = np.array([w, h])
                    w_z = size[0] + self.context_amount * np.sum(size)
                    h_z = size[1] + self.context_amount * np.sum(size)
                    s_z = round(np.sqrt(w_z * h_z))

                    if self.is_search_local:
                        # if doing local search, then we start from the position of
                        # the object in the previous frame, and will update the both
                        # position and size online
                        self.target_position = xywh_2_cxywh(target_bbox)[:2]
                        self.target_size = target_bbox[2:]
                    else:
                        # If doing global search (search the entire image), then we need
                        # to fix the position at the center of image and search_size to the size
                        # of the entire image. We will/should NOT update these parameters.
                        self.target_position = np.array([img_w // 2, img_h // 2])
                        self.target_size = target_bbox[2:]

                    # Crop template region from the whole image.
                    avg_chans = np.mean(template.image, axis=(0, 1))
                    template_crop = get_subwindow(
                        template.image,
                        pos=pos,
                        original_sz=s_z,
                        model_sz=self.template_size,
                        avg_chans=avg_chans,
                    )

                # Extract template feature.
                template_crop = template_crop.unsqueeze(dim=0).to(self.device)
                template_feature = self.model.feature_extractor(template_crop)

                # Store extracted feature in registration.
                template.update_template("features", template_feature.data.cpu())

                logging.info(
                    "Updated template feature for instance id: {}, template idx: {}".format(
                        instance_id, idx
                    )
                )

    def run_model(
        self,
        img: torch.Tensor,
    ) -> Dict:
        """
        Run one step of inference from a given search image.

        Args:
            img: search image in cv2 format.
            multiscale_list: a list that stores different scales used for SiamMask.
        Returns:
            results: a dictionary holding tracking results of all targets for one image
        """
        # Return empty query result if no target.
        if len(self.instance_templates) == 0:
            return {}

        img = img.permute(0, 2, 3, 1).contiguous().numpy()[0]

        img_h, img_w, _ = img.shape
        pos = self.target_position
        target_size = self.target_size
        wc_x = target_size[1] + self.context_amount * np.sum(target_size)
        hc_x = target_size[0] + self.context_amount * np.sum(target_size)
        s_x = np.sqrt(wc_x * hc_x)
        scale_x = self.template_size / s_x
        global_search_needs_crop = False

        if self.is_search_local:
            d_search = (self.search_size - self.template_size) / 2
            pad = d_search / scale_x
            # s_search = s_x * (self.search_size) / (self.template_size)
            s_search = round(s_x + 2 * pad)

            logging.info(
                f"is_search_local {self.is_search_local}, target position {pos}, target size {target_size}, template_size {self.template_size}, search_size {self.search_size}, s_search {s_search}."
            )
        else:
            img_size = max(img_w, img_h)
            # Need to take into account pixels around the image border
            search_size = img_size * scale_x

            # If the global search size is larger than the global_search_max_search_size,
            # we need to crop the input into smaller patches and feed one at a time.
            # Otherwise, we will be out of GPU memory if the search_size is large
            if search_size > self.search_size:
                global_search_needs_crop = True
                maximum_input_size = self.search_size
                overlap_size = (
                    s_x  # crop overlap to make sure object do not occur at the border
                )

                maximum_input_original_image = maximum_input_size / scale_x
                # need to make sure, for each crop, we give it enough pad to match target_size
                effective_input_original = maximum_input_original_image - overlap_size
                n_per_x = int(img_w // effective_input_original) + 1
                n_per_y = int(img_h // effective_input_original) + 1
                line_x = (
                    np.arange(0, img_w, effective_input_original)
                    + effective_input_original // 2
                )  # convert to center
                line_y = (
                    np.arange(0, img_h, effective_input_original)
                    + effective_input_original // 2
                )
                x_centers, y_centers = np.meshgrid(line_x, line_y)
                x_centers = x_centers.flatten()
                y_centers = y_centers.flatten()

                s_search = maximum_input_original_image
                search_size = self.search_size
                logging.info(
                    f"Global search needs crop, template original size {s_x}, scale {scale_x}, search image entire size {search_size}, "
                )
                logging.info(
                    f"Global search needs crop, maximum_input_original_image, {maximum_input_original_image}, effective_input_original, {effective_input_original}, n_x {n_per_x}, n_y {n_per_y}"
                )

            else:
                global_search_needs_crop = False

                d_search = (self.search_size - self.template_size) / 2
                pad = d_search / scale_x
                # s_search = s_x * (self.search_size) / (self.template_size)
                s_search = round(s_x + 2 * pad)

                logging.info(
                    f"Global search target position {pos}, target size {target_size}, template_size {self.template_size}, search_size {self.search_size}, s_search {s_search}."
                )

        # TODO: Remove Multi-scale inference.
        # TODO: Remove multi-template selection as well.
        multiscale_results = []
        avg_chans = np.mean(img, axis=(0, 1))
        for scale in self.multiscale_list:
            assert scale >= 1.0, "Only support scale >= 1.0 for now."

            if global_search_needs_crop:
                # Resize once, so we do not need to resize multiple times in get_subwindow
                resized_img = cv2.resize(
                    img, (int(img_w * scale_x), int(img_h * scale_x))
                )

                crops = []
                for x, y in zip(x_centers, y_centers):
                    search_crop = get_subwindow(
                        resized_img,
                        [x * scale_x, y * scale_x],
                        maximum_input_size,
                        maximum_input_size,
                        avg_chans,
                    )
                    crops.append(search_crop)
                results = []

                # Initilize our search for which crop has the highest response
                best_score_idx = 0
                best_score = 0
                test_batch = self.test_batch_size
                n_batch = int(math.ceil(len(crops) / test_batch))
                for i in range(n_batch):
                    search_crop = crops[i * test_batch : (i + 1) * test_batch]
                    search_crop = torch.stack(search_crop)
                    search_crop = search_crop.to(self.device)
                    search_feature = self.model.feature_extractor(
                        search_crop, save_fm=True
                    )

                    # Run siammask matching.
                    res = self._siammask_query(search_feature, enable_mask=False)

                    # TODO: remove multiple templates, multiple instances, only 1 instance 1 template supported
                    scores, delta, mask = res[0][0]
                    total, _ = scores.shape

                    for j in range(total):
                        max_s = np.max(scores[j])
                        if max_s > best_score:
                            best_score = max_s
                            best_score_idx = i * test_batch + j
                        res = [
                            [
                                [
                                    scores[j],
                                    delta[j],
                                    mask[j] if mask is not None else None,
                                ]
                            ]
                        ]
                        results.append(res)

                idx = best_score_idx
                best_crop_res = results[idx]
                results = best_crop_res
                results = self._select_template(results, mode="best")
                pos = [x_centers[idx], y_centers[idx]]

                i = idx // n_per_x
                j = idx % n_per_x
                logging.info(
                    f"Global search needs crop, best crop idx {idx}, patch i, j {i}, {j}"
                )
            else:
                # Search region will be resized to search_size.
                search_size = int(self.search_size * scale)

                # Extract search region.
                search_crop = get_subwindow(
                    img,
                    pos,
                    s_search,
                    search_size,
                    avg_chans,
                )
                # Extract convolutional feature for the search region.
                search_crop = search_crop.unsqueeze(dim=0).to(self.device)
                search_feature = self.model.feature_extractor(search_crop, save_fm=True)

                # Run siammask matching.
                # TODO: remove multi-template and multi-scale
                results = self._siammask_query(
                    search_feature, enable_mask=self.enable_mask
                )
                scores, delta, mask = results[0][0]

                results = [
                    [[scores[0], delta[0], mask[0] if mask is not None else None]]
                ]

                # Select the best template with the highest matching score.
                results = self._select_template(results, mode="best")

            multiscale_results.append(results)

        # Select the best scale with the highest matching score.
        best_results, best_scores = self._select_multiscale(
            multiscale_results, self.multiscale_list
        )

        # Post-process the best result.
        postprocessed_results = self._post_process(
            best_results,
            best_scores,
            pos,
            scale_x,
            s_search,
            img_w,
            img_h,
            enable_refine=self.enable_refine,
        )

        return postprocessed_results

    def update_tracker(self, results):
        assert len(results) == 1, "Only support tracking single object"

        for _, res in results.items():
            bbox = res["bbox"]
            # Convert to center xy
            cbox = xywh_2_cxywh(bbox)[:2]
            # Update the pos and size only when we are tracking the object locally
            if self.is_search_local:
                self.target_position = cbox[:2]
                # Ensure a minimum and maximum target bbox size
                self.target_position[0] = max(
                    0, min(res["img_w"], self.target_position[0])
                )
                self.target_position[1] = max(
                    0, min(res["img_h"], self.target_position[1])
                )

                self.target_size = bbox[2:]
                self.target_size[0] = max(10, min(res["img_w"], self.target_size[0]))
                self.target_size[1] = max(10, min(res["img_h"], self.target_size[1]))

    def _siammask_query(self, search_feature: Any, enable_mask: bool = False) -> List:
        """
        Run SiamMask matching and headers to get query results.

        Args:
            search_feature: extracted convolutional feature of search image
            enable_mask: whether return object mask
        Return:
            results: [
                # instance1
                [
                    # template1
                    (score, delta),
                    #template2
                    (score, delta),
                ],
                # instance2
                [
                    ...
                ]
            ]
        """
        # Query all instances if self.query_list is None.
        query_list = self.instance_templates.get_instance_ids()

        results = []
        # Loop for all query instances.
        for instance_id in query_list:
            temp_results = []
            try:
                instance = self.instance_templates.get_instance(instance_id)
            except Exception:
                raise ValueError(
                    "Instance_id {} has not registered yet!".format(instance_id)
                )

            # Loop for all templates of the instance.
            for template in instance:
                # TODO: features are in cpu for now, figure out a more efficient way later.
                # print(search_feature.shape)
                template_feature = template.features
                _, c, h, w = template_feature.shape
                b, _, _, _ = search_feature.shape
                template_feature = template_feature.expand(b, c, h, w)

                score, delta, mask = self.model.query(
                    template_feature.to(self.device),
                    search_feature,
                    enable_mask=enable_mask,
                )
                if enable_mask:
                    mask = mask[0].sigmoid().cpu().data.numpy()

                score = (
                    F.softmax(
                        score.view(b, 2, -1),
                        dim=1,
                    )
                    .data[:, 1, :]
                    .cpu()
                    .numpy()
                )
                # delta = delta.permute(1, 2, 3, 0).contiguous().view(4, b, -1).permute(1, 0, 2).data.cpu().numpy()
                delta = delta.view(b, 4, -1).data.cpu().numpy()

                temp_results.append([score, delta, mask])

            results.append(temp_results)
        return results

    def _select_template(self, results, mode="best"):
        """
        Comebine / select results from multiple templates of an instance.
        """
        selected_results = []

        for instance_result in results:
            if len(instance_result) == 1:
                selected_results.append(instance_result[0])
            else:
                # TODO: add at least mode=="best", select the result with best score.
                raise NotImplementedError

        return selected_results

    def _select_multiscale(self, multiscale_results, multiscale_list):
        """
        Comebine / select query results of multiple scales according to the matching scores.
        """
        if len(multiscale_list) == 1:
            best_results = [
                [multiscale_list[0]] + result for result in multiscale_results[0]
            ]
            best_scores = [
                np.max(result[1]) for result in best_results
            ]  # [scale, score, delta] for each result
            return best_results, best_scores

        # Store best results for all instances
        best_results = [None for _ in range(len(multiscale_results[0]))]
        best_scores = [0.0 for _ in range(len(multiscale_results[0]))]

        # Loop for all scales.
        for results, scale in zip(multiscale_results, multiscale_list):
            # Loop for all instances.
            for i, result in enumerate(results):
                score, delta, mask = result
                best_pscore = np.max(score)

                if best_pscore > best_scores[i]:
                    best_results[i] = [scale, score, delta, mask]
                    best_scores[i] = best_pscore

        return best_results, best_scores

    def _post_process_mask(
        self, scale, mask, best_pscore_id, score_size, img_w, img_h, enable_refine=True
    ):
        """
        Post processing of the mask from the query result.

        This function post-processes mask by selecting the mask from the location indicated by
        best_pscore_id, refine the mask if enable_refine and then convert back to the original
        image coordinate.
        """
        best_pscore_id_mask = np.unravel_index(
            best_pscore_id, (5, score_size, score_size)
        )
        delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]

        if enable_refine:
            mask = (
                self.model.track_refine((delta_y, delta_x))
                .sigmoid()
                .squeeze()
                .view(self.out_size, self.out_size)
                .cpu()
                .data.numpy()
            )
        else:
            mask = mask[:, delta_y, delta_x]
            mask = np.squeeze(mask)
            mask = np.reshape(mask, (self.out_size, self.out_size))

        def crop_back(image, bbox, out_sz, padding=-1):
            a = (out_sz[0] - 1) / bbox[2]
            b = (out_sz[1] - 1) / bbox[3]
            c = -a * bbox[0]
            d = -b * bbox[1]
            mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)
            crop = cv2.warpAffine(
                image,
                mapping,
                (out_sz[0], out_sz[1]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=padding,
            )
            return crop

        # Because we use the whole image as search space (instead the target center),
        # we set the target_pos to be the center of the search image
        target_pos = [img_w // 2, img_h // 2]
        s_x = max(img_h, img_w) + self.template_size // 2

        crop_box = [
            target_pos[0] - round(s_x) / 2,
            target_pos[1] - round(s_x) / 2,
            round(s_x),
            round(s_x),
        ]

        s = crop_box[2] / self.search_size / scale
        sub_box = [
            crop_box[0] + (delta_x - self.base_size / 2) * self.total_stride * s,
            crop_box[1] + (delta_y - self.base_size / 2) * self.total_stride * s,
            s * self.template_size,
            s * self.template_size,
        ]
        s = self.out_size / sub_box[2]
        back_box = [
            -sub_box[0] * s,
            -sub_box[1] * s,
            img_w * s,
            img_h * s,
        ]
        mask_in_img = crop_back(mask, back_box, (img_w, img_h))
        target_mask = (mask_in_img > self.seg_thr).astype(np.uint8)

        return target_mask

    def _post_process(
        self,
        best_results,
        best_scores,
        pos,
        scale_x,
        s_search,
        img_w,
        img_h,
        enable_refine=False,
    ):
        """
        Post processing of the query result.

        This function post-processes the best results, including filtering low-score results,
        transforming delta to bbox ([xywh] format).
        """
        # Query all instances if self.query_list is None.
        query_list = self.instance_templates.get_instance_ids()

        query_results = {}
        for i, instance_id in enumerate(query_list):
            if best_scores[i] > self.confidence_thresh:
                scale, score, delta, mask = best_results[i]

                search_size = int(self.search_size * scale)
                scale_z = search_size / s_search
                score_size = (
                    (search_size - self.template_size) // self.model.total_stride
                    + 1
                    + self.model.base_size
                )
                anchor = generate_anchor(self.model.anchors, score_size)

                delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
                delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
                delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
                delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]

                target_sz = np.array(self.target_size)

                def change(r):
                    return np.maximum(r, 1.0 / r)

                def sz(w, h):
                    pad = (w + h) * 0.5
                    sz2 = (w + pad) * (h + pad)
                    return np.sqrt(sz2)

                def sz_wh(wh):
                    pad = (wh[0] + wh[1]) * 0.5
                    sz2 = (wh[0] + pad) * (wh[1] + pad)
                    return np.sqrt(sz2)

                target_sz_in_crop = target_sz * scale_x

                if self.is_search_local:
                    if self.windowing == "cosine":
                        window = np.outer(
                            np.hanning(score_size), np.hanning(score_size)
                        )
                    elif self.windowing == "uniform":
                        window = np.ones((score_size, score_size))
                    else:
                        raise NotImplementedError
                    window = np.tile(window.flatten(), self.anchor_num)
                    # If search local, need to apply motion model to penalize large motion
                    # smooth the location and size change
                    s_c = change(
                        sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop))
                    )  # scale penalty
                    r_c = change(
                        (target_sz_in_crop[0] / target_sz_in_crop[1])
                        / (delta[2, :] / delta[3, :])
                    )  # ratio penalty

                    penalty = np.exp(-(r_c * s_c - 1) * self.penalty_k)
                    pscore = penalty * score

                    # cos window (motion model)
                    pscore = (
                        pscore * (1 - self.window_influence)
                        + window * self.window_influence
                    )
                    best_pscore_id = np.argmax(pscore)
                    lr = (
                        penalty[best_pscore_id] * score[best_pscore_id] * self.lr
                    )  # lr for OTB
                    pred_in_crop = delta[:, best_pscore_id] / scale_x

                    res_x = pred_in_crop[0] + pos[0]
                    res_y = pred_in_crop[1] + pos[1]
                    res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
                    res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr
                else:
                    best_pscore_id = np.argmax(score)

                    pred_in_crop = delta[:, best_pscore_id] / scale_x

                    res_x = pred_in_crop[0] + pos[0]
                    res_y = pred_in_crop[1] + pos[1]
                    res_w = pred_in_crop[2]
                    res_h = pred_in_crop[3]

                # bbox is stored in [xywh] format (xy is the top-left corner)
                bbox_pred = cxywh_2_xywh([res_x, res_y, res_w, res_h])

                # convert to float for json compatibility
                bbox_pred_float = [float(num) for num in bbox_pred]

                # if enable_mask
                if mask is not None:
                    mask = self._post_process_mask(
                        scale,
                        mask,
                        best_pscore_id,
                        score_size,
                        img_w,
                        img_h,
                        enable_refine=enable_refine,
                    )

                # Record query results.
                query_results[instance_id] = {
                    "score": float(score[best_pscore_id]),
                    "bbox": bbox_pred_float,
                    "mask": mask,
                    "img_w": img_w,
                    "img_h": img_h,
                }

        return query_results
