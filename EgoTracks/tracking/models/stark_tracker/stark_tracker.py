import logging
import math
import time
from copy import deepcopy
from typing import Dict, List

import cv2
import numpy as np
import torch
from fvcore.common.config import CfgNode
from tracking.models.single_object_tracker import SingleObjectTracker
from tracking.models.stark_tracker.stark_st import build_starkst
from tracking.models.stark_tracker.utils.box_ops import clip_box
from tracking.models.stark_tracker.utils.merge import merge_template_search
from tracking.models.stark_tracker.utils.misc import NestedTensor, Preprocessor
from tracking.models.stark_tracker.utils.preprocessing_utils import sample_target
from tracking.utils.bbox_helper import xywh_2_cxywh
from tracking.utils.env import pathmgr


class STARKTracker(SingleObjectTracker):
    def __init__(
        self,
        cfg: CfgNode,
        device: torch.device = None,
        verbose: bool = False,
    ):
        # Define siammask model
        # Override search size to correctly initilize the box head
        cfg.defrost()
        cfg.DATA.SEARCH.SIZE = cfg.TEST.SEARCH_SIZE
        cfg.freeze()
        logging.info(cfg)
        pretrained_path = cfg.MODEL.WEIGHTS
        # If the input image needs to be cropped to smaller patches,
        # this variable controls how many these patches we can send at a time.
        self.cfg = cfg

        self.template_factor: float = cfg.TEST.TEMPLATE_FACTOR
        self.template_size: int = cfg.TEST.TEMPLATE_SIZE
        self.search_size: int = cfg.TEST.SEARCH_SIZE
        self.context_amount: float = 1.0 / self.template_factor
        self.update_intervals: List = cfg.TEST.UPDATE_INTERVALS.UPDATE_INTERVALS
        self.num_extra_template = len(self.update_intervals)
        self.test_batchsize = cfg.TEST.TEST_BATCHSIZE
        # template update
        self.z_dict1 = {}
        self.z_dict_list = []
        self.preprocessor = Preprocessor()
        self.frame_step = 0
        self.target_id = None
        self.device = device if device is not None else torch.device("cpu")

        stark_st = build_starkst(cfg)

        if pretrained_path is not None:
            logging.info(f"Loading pretrained model from {pretrained_path} ...")
            with pathmgr.open(pretrained_path, "rb") as f:
                stark_st.load_state_dict(torch.load(f)["state_dict"], strict=True)
        super(STARKTracker, self).__init__(stark_st, verbose)

        self.model = self.model.to(self.device)

        # This was put at the end to override parent function
        self.is_search_local: bool = cfg.TEST.IS_SEARCH_LOCAL

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
        # make sure the size of the box is at least 10
        # 10 is also the margin used in the predicted bbox
        target_bbox[2] = max(target_bbox[2], 10)
        target_bbox[3] = max(target_bbox[3], 10)

        if isinstance(img, torch.Tensor):
            img_npy = img.permute(0, 2, 3, 1).contiguous().numpy()[0]
        elif isinstance(img, np.ndarray):
            img_npy = img
        else:
            raise NotImplementedError
        img_h, img_w, _ = img_npy.shape
        # initialize z_dict_list
        self.z_dict_list = []

        # get the 1st template
        # calculate the crop_size; adapted from SiamMask
        x, y, w, h = xywh_2_cxywh(target_bbox)
        pos = np.array([x, y])
        size = np.array([w, h])
        w_z = size[0]
        h_z = size[1]
        s_z = round(np.math.ceil(np.sqrt(w_z * h_z) / self.context_amount))

        z_patch_arr1, _, z_amask_arr1 = sample_target(
            img_npy,
            pos,
            s_z,
            output_sz=self.template_size,
        )
        template1 = self.preprocessor.process(
            z_patch_arr1,
            z_amask_arr1,
            device=self.device,
            verbose=self.verbose,
        )
        with torch.no_grad():
            self.z_dict1 = self.model.forward_backbone(template1)
        # get the complete z_dict_list
        self.z_dict_list.append(self.z_dict1)
        for _ in range(self.num_extra_template):
            self.z_dict_list.append(deepcopy(self.z_dict1))

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
        self.frame_step = 0
        self.target_id = target_id

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
            {
                "target_id": {
                    "bbox": [x, y, w, h],
                    "score": float,
                }
            }
        """
        self.frame_step += 1

        if self.is_search_local:
            return self.track_locally(img)
        else:
            return self.track_globally(img)

    def track_globally(
        self,
        img: torch.Tensor,
    ) -> Dict:
        img_npy = img.permute(0, 2, 3, 1).contiguous().numpy()[0]

        img_h, img_w, _ = img_npy.shape
        pos = self.target_position
        target_size = self.target_size
        # TODO: I believe the way they implement this is following SiamFC/SiamRPN/SiamRCNN. However, the search_size
        # here may not always match the template scale. Will need to modify it according to our previous implementation
        # on SiamMask

        # calculate the resize_factor
        wc_x = target_size[1]
        hc_x = target_size[0]
        s_x = round(np.math.ceil(np.sqrt(wc_x * hc_x) / self.context_amount))
        scale_x = self.template_size / s_x

        img_size = max(img_w, img_h)
        # Need to take into account pixels around the image border
        search_size = img_size * scale_x

        # If the global search size is larger than the global_search_max_search_size,
        # we need to crop the input into smaller patches and feed one at a time.
        # Otherwise, we will be out of GPU memory if the search_size is large
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

        search_size = self.search_size
        logging.info(
            f"Global search needs crop, template original size {s_x}, scale {scale_x}, search image entire size {search_size}, "
        )
        logging.info(
            f"Global search needs crop, maximum_input_original_image, {maximum_input_original_image}, effective_input_original, {effective_input_original}, n_x {n_per_x}, n_y {n_per_y}"
        )

        # Resize once, so we do not need to resize multiple times in get_subwindow
        resized_img = cv2.resize(img_npy, (int(img_w * scale_x), int(img_h * scale_x)))
        resize_factor = scale_x

        crops = []
        for x, y in zip(x_centers, y_centers):
            x_patch_arr, _, x_amask_arr = sample_target(
                resized_img,
                # need to convert cx, cy, w, h to x, y, w, h
                [x * scale_x, y * scale_x],
                self.search_size,
                output_sz=self.search_size,
                verbose=self.verbose,
            )  # (x1, y1, w, h)

            search = self.preprocessor.process(x_patch_arr, x_amask_arr).to(self.device)
            crops.append(search)
        all_scores = []
        all_bboxes = []

        test_batch = self.test_batchsize
        n_batch = int(math.ceil(len(crops) / test_batch))
        for i in range(n_batch):
            search_crop = crops[i * test_batch : (i + 1) * test_batch]
            batchsize = len(search_crop)
            tensors = [s.tensors.squeeze() for s in search_crop]
            mask = [s.mask.squeeze() for s in search_crop]
            tensors = torch.stack(tensors)
            mask = torch.stack(mask)
            search_crop = NestedTensor(tensors, mask)

            z_dict_list = []
            for z in self.z_dict_list:
                n, _, c = z["feat"].shape
                feat = z["feat"].expand(n, batchsize, c)
                _, c = z["mask"].shape
                mask = z["mask"].expand(batchsize, c)
                n, _, c = z["pos"].shape
                pos = z["pos"].expand(n, batchsize, c)

                z_dict_list.append({"feat": feat, "mask": mask, "pos": pos})

            with torch.no_grad():
                x_dict = self.model.forward_backbone(search_crop)

                feat_dict_list = z_dict_list + [x_dict]
                seq_dict = merge_template_search(feat_dict_list)
                # run the transformer
                out_dict, _, _ = self.model.forward_transformer(
                    seq_dict=seq_dict, run_box_head=True, run_cls_head=True
                )

            # get the final result
            pred_boxes = out_dict["pred_boxes"].view(-1, 4)
            conf_score = out_dict["pred_logits"].view(-1).sigmoid()

            total = len(conf_score)
            for i in range(total):
                all_scores.append(conf_score[i].item())
                all_bboxes.append(
                    (pred_boxes[i] * self.search_size / resize_factor).tolist()
                )

        idx = np.argmax(all_scores)
        i = idx // n_per_x
        j = idx % n_per_x
        pos = [x_centers[idx], y_centers[idx]]
        pred_box = all_bboxes[idx]
        pred_box = clip_box(
            self.map_box_back(pred_box, resize_factor, pos=pos), img_h, img_w, margin=10
        )
        conf_score = all_scores[idx]
        logging.info(
            f"Global search needs crop, best crop idx {idx}, patch i, j {i}, {j}"
        )

        return {
            self.target_id: {
                "bbox": pred_box,
                "score": conf_score,
                "img_npy": img_npy,
            }
        }

    def track_locally(
        self,
        img: torch.Tensor,
    ) -> Dict:
        if isinstance(img, torch.Tensor):
            img_npy = img.permute(0, 2, 3, 1).contiguous().numpy()[0]
        elif isinstance(img, np.ndarray):
            img_npy = img
        else:
            raise NotImplementedError

        img_h, img_w, _ = img_npy.shape
        pos = self.target_position
        target_size = self.target_size
        # TODO: I believe the way they implement this is following SiamFC/SiamRPN/SiamRCNN. However, the search_size
        # here may not always match the template scale. Will need to modify it according to our previous implementation
        # on SiamMask

        # calculate the resize_factor
        wc_x = target_size[1]
        hc_x = target_size[0]
        s_x = round(math.ceil(np.sqrt(wc_x * hc_x) / self.context_amount))

        crop_sz = round(math.ceil(s_x * (self.search_size) / (self.template_size)))

        t = time.time()
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            img_npy,
            # need to convert cx, cy, w, h to x, y, w, h
            pos,
            crop_sz,
            output_sz=self.search_size,
            verbose=self.verbose,
        )  # (x1, y1, w, h)

        if self.verbose:
            logging.error(
                f"Crop patch size {crop_sz} wc_x hc_x {(wc_x, hc_x)} img size {(img_h, img_w)} {time.time() - t}"
            )

        # There might be slightly rounding errors, but the difference should be fairly small
        # assert abs(scale_x - resize_factor) <= 0.01, f"scale_x: {scale_x}, resize_factor: {resize_factor}"

        t = time.time()
        search = self.preprocessor.process(x_patch_arr, x_amask_arr, device=self.device)

        if self.verbose:
            logging.error(
                f"Preprocess image with size {x_patch_arr.shape} takes {time.time() - t}"
            )

        with torch.no_grad():
            t = time.time()
            x_dict = self.model.forward_backbone(search)

            if self.verbose:
                logging.error(f"Feature extraction {time.time() - t}")
            # merge the template and the search
            feat_dict_list = self.z_dict_list + [x_dict]
            seq_dict = merge_template_search(feat_dict_list)
            # run the transformer
            t = time.time()
            out_dict, _, _ = self.model.forward_transformer(
                seq_dict=seq_dict, run_box_head=True, run_cls_head=True
            )
            if self.verbose:
                logging.error(
                    f"Matching transformer {time.time() - t}"
                )  # get the final result
        pred_boxes = out_dict["pred_boxes"].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (
            pred_boxes.mean(dim=0) * self.search_size / resize_factor
        ).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        pred_box = clip_box(
            self.map_box_back(pred_box, resize_factor), img_h, img_w, margin=10
        )
        # get confidence score (whether the search region is reliable)
        conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()

        return {
            self.target_id: {
                "bbox": pred_box,
                "score": conf_score,
                "img_npy": img_npy,
            }
        }

    def update_tracker(self, results):
        assert len(results) == 1, "Only support tracking single object"

        for _, res in results.items():
            bbox = res["bbox"]
            score = res["score"]
            image = res["img_npy"]
            # Convert to center xy
            cbox = xywh_2_cxywh(bbox)[:2]
            # Update the pos and size only when we are tracking the object locally
            if self.is_search_local:
                self.target_position = cbox[:2]
                self.target_size = bbox[2:]

                # update template
                for idx, update_i in enumerate(self.update_intervals):
                    if self.frame_step % update_i == 0 and score > 0.5:
                        # calculate crop_sz
                        pos = self.target_position
                        target_size = self.target_size
                        wc_x = target_size[1]
                        hc_x = target_size[0]
                        s_x = round(
                            math.ceil(np.sqrt(wc_x * hc_x) / self.context_amount)
                        )

                        z_patch_arr, _, z_amask_arr = sample_target(
                            image,
                            pos,
                            s_x,
                            output_sz=self.template_size,
                            verbose=self.verbose,
                        )  # (x1, y1, w, h)
                        template_t = self.preprocessor.process(
                            z_patch_arr, z_amask_arr, device=self.device
                        )
                        with torch.no_grad():
                            z_dict_t = self.model.forward_backbone(template_t)
                        self.z_dict_list[
                            idx + 1
                        ] = z_dict_t  # the 1st element of z_dict_list is template from the 1st frame
                        logging.info(
                            f"Update template {idx + 1}, confidence score {score}"
                        )

    def map_box_back(self, pred_box: List, resize_factor: float, pos: List = None):
        if pos is None:
            cx_prev, cy_prev = (
                self.target_position[0],
                self.target_position[1],
            )
        else:
            cx_prev, cy_prev = pos

        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]
