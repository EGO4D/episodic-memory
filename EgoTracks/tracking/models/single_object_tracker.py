import logging
import time
from abc import abstractmethod
from typing import Dict

import av
import torch
from tqdm import tqdm
from tracking.dataset.eval_datasets.base_dataset import Sequence
from tracking.models.template import InstanceRegistration
from tracking.models.tracker import Tracker
from tracking.utils.utils import opencv_loader


class SingleObjectTracker(Tracker):
    """
    Define a base class for single object tracker (SOT).

    In SOT, the bounding box of the target object is given in the first frame.
    """

    def __init__(self, model, verbose: bool = False):
        super().__init__(model, verbose)
        self.instance_templates = InstanceRegistration()

        # By default, SOT starts with search local region
        self.is_search_local = True
        self.target_position = None
        self.target_size = None

    def inference_video_handler(
        self, video: av.container.Container, meta_data: Dict = None
    ) -> Dict:
        raise NotImplementedError

    def inference_sequence(self, video: Sequence, meta_data: Dict = None) -> Dict:
        """
        Run tracking model on a video sequence. Since some dataset stores video as individual frames and each sequence
        can be quite large, it is not possible to pre-fetch all the data into memory (especially in multi-processing). So
        we fetch those frames in an online fashion.

        Args:
            video: a Sequence contains frames path for the sequence
            meta_data: contains any other information regarding the specific video.
                It should contain:
                "target_bbox", bounding box [x, y, w, h]
                "target_id", target_id is a string and identifies a unique instance
                "frame_numbers", corresponds to the frame number of each input image in the original video
        Returns:
            dict: contains tracking results, where key is the object_id. An example of the return result
                should look like:
                {
                    "object_id_1":{
                        "label": str,
                        "bboxes": [
                            {
                                "frame_number": int,
                                "bbox": [x, y, w, h],
                                "score": float (confidence score),
                            },
                            {
                                "frame_number": int,
                                "bbox": [x, y, w, h],
                                "score": float (confidence score),
                            },
                            ...
                        ]
                }
        """
        self.model.eval()

        target_bbox = meta_data["target_bbox"]
        target_id = meta_data["target_id"]
        frame_numbers = meta_data["frame_numbers"]
        result = {target_id: {"label": target_id, "bboxes": []}}
        for i in tqdm(range(len(frame_numbers)), total=len(frame_numbers)):
            img_path = video.frames[frame_numbers[i]]
            img = opencv_loader(img_path)
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            if i == 0:
                # the first frame of the video, initilize tracker
                self.init_tracker(img, meta_data)

                result[target_id]["bboxes"].append(
                    {
                        "bbox": list(target_bbox),
                        "score": 1,
                        "frame_number": frame_numbers[i],
                        "type": "gt",
                    }
                )
                # In case not start at the frame next to the template,
                # we may want to init the tracker from the another location
                if "init_location" in meta_data:
                    self.target_position = meta_data["init_location"]

                if "init_size" in meta_data:
                    self.target_size = meta_data["init_size"]
            else:
                results = self.run_model(img)
                assert len(results) == 1, "Only support single object tracking!"
                self.update_tracker(results)
                for target_id, res in results.items():
                    bbox = res["bbox"]
                    score = res["score"]
                    result[target_id]["bboxes"].append(
                        {
                            "bbox": bbox,
                            "score": score,
                            "frame_number": frame_numbers[i],
                            "type": "prediction",
                        }
                    )

        self.reset_tracker()
        return result

    def inference_img_sequence(
        self, video: torch.Tensor, meta_data: Dict = None
    ) -> Dict:
        """
        Run tracking model on a video sequence.

        Args:
            video: a torch Tensor that contains a sequence of images, [N, 3, H, W]
            meta_data: contains any other information regarding the specific video.
                It should contain:
                "target_bbox", bounding box [x, y, w, h]
                "target_id", target_id is a string and identifies a unique instance
                "frame_numbers", corresponds to the frame number of each input image in the original video
        Returns:
            dict: contains tracking results, where key is the object_id. An example of the return result
                should look like:
                {
                    "object_id_1":{
                        "label": str,
                        "bboxes": [
                            {
                                "frame_number": int,
                                "bbox": [x, y, w, h],
                                "score": float (confidence score),
                            },
                            {
                                "frame_number": int,
                                "bbox": [x, y, w, h],
                                "score": float (confidence score),
                            },
                            ...
                        ]
                }
        """
        assert "target_bbox" in meta_data
        assert "target_id" in meta_data
        assert "frame_numbers" in meta_data
        self.model.eval()

        target_bbox = meta_data["target_bbox"]
        target_id = meta_data["target_id"]
        frame_numbers = meta_data["frame_numbers"]
        result = {target_id: {"label": target_id, "bboxes": []}}
        # _, _, h, w = video.shape
        for i in range(len(video)):
            if isinstance(video, torch.Tensor):
                img = video[i].unsqueeze(0)
            else:
                img = video[i]
            if i == 0:
                # the first frame of the video, initilize tracker
                t = time.time()
                self.init_tracker(img, meta_data)
                if self.verbose:
                    logging.error(f"Init tracker: {time.time() - t}")

                result[target_id]["bboxes"].append(
                    {
                        "bbox": list(target_bbox),
                        "score": 1,
                        "frame_number": frame_numbers[i],
                        "type": "gt",
                    }
                )
            else:
                results = self.run_model(img)
                assert len(results) == 1, "Only support single object tracking!"
                self.update_tracker(results)
                for target_id, res in results.items():
                    bbox = res["bbox"]
                    score = res["score"]
                    result[target_id]["bboxes"].append(
                        {
                            "bbox": bbox,
                            "score": score,
                            "frame_number": frame_numbers[i],
                            "type": "prediction",
                        }
                    )

        self.reset_tracker()
        return result

    @abstractmethod
    def run_model(self, image: torch.Tensor) -> Dict:
        """
        Most tracking models run frame by frame. This function runs a single forward of the tracking model.

        Args:
            image: this is one image where we run one step of tracking, [N, 3, H, W]
                N is the batchsize. In inference mode, N should equal to 1.
        Returns:
            dict: a dictionary holding tracking results for one image.
                The structure of the dict is not listed as it may be specific to the model.
                Common result contains "bbox" and "score"
        """
        pass

    @abstractmethod
    def update_tracker(self, result: Dict) -> None:
        """
        This function is used to update tracking information as the model runs through the video.
        In SOT, the most important update is to update the local search window.

        Args:
            results: a dictionary holding tracking results of for one image. It is designed to directly
                use the output from self.run_model(image).
        Returns:
            None
        """
        pass

    @abstractmethod
    def init_tracker(self, image: torch.Tensor, meta: Dict) -> None:
        """
        This function is used to initilize SOT with first frame annotation

        Args:
            image: this is one image where we used to initilize the tracker, [1, 3, H, W]
            meta: a dictionary contains information of target objects in this image.
        Returns:
            None
        """
        pass

    def reset_tracker(self) -> None:
        """
        To reset the tracker after done running on one video.
        Clear any intermediate variables to get a clean setup for next video.
        """
        self.instance_templates = InstanceRegistration()
        self.target_position = None
