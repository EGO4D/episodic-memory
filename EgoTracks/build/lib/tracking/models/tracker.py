from abc import abstractmethod
from typing import Dict, Union

import av
import numpy as np
import torch
from torch import nn
from tracking.dataset.eval_datasets.base_dataset import Sequence


class Tracker(nn.Module):
    """
    Define a base class for tracking models.

    The base class should have common functions designed for both
    single object tracking (SOT) and multiple object tracking (MOT)
    """

    def __init__(self, model: nn.Module, verbose: bool = False):
        """
        Initialize a Tracker object.

        Args:
            model: a tracking model
            verbose: boolean flag that indicates whether to log profiling information.
        """
        super().__init__()
        self.model = model
        self.verbose = verbose

    @abstractmethod
    def inference(
        self,
        video: Union[torch.Tensor, np.ndarray, av.container.Container, Sequence],
        meta_data: Dict = None,
    ) -> Dict:
        """
        Run tracking model on a video sequence.

        Args:
            video: either a torch Tensor that contains a sequence of images, [N, 3, H, W]
                or a pointer to the video file (typically .mp4). Currently we only support
                torch.Tensor as input.

                The torch.Tensor is intended for short videos or dataset that consists of pngs,
                while the av.container.Container is meant for long videos where loading each frame
                into the memory is not practical.
            meta_data: contains any other information regarding the specific video.
                Examples of information can be stored are as follows:
                For SOT, it should countain information such as initial target bbox.
                For long videos, it could contain start and end frame number or timestamp where we
                want to run tracking model.
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
                    },
                    "object_id_2":{},
                    ...
                }
        """
        if isinstance(video, torch.Tensor) or isinstance(video, np.ndarray):
            result = self.inference_img_sequence(video, meta_data)
        elif isinstance(video, av.container.Container):
            result = self.inference_video_handler(video, meta_data)
        elif isinstance(video, Sequence):
            result = self.inference_sequence(video, meta_data)
        else:
            raise NotImplementedError

        return result

    @abstractmethod
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

    @abstractmethod
    def inference_img_sequence(
        self, video: torch.Tensor, meta_data: Dict = None
    ) -> Dict:
        """
        Run tracking model on a video sequence.

        Args:
            video: torch Tensor that contains a sequence of images, [N, 3, H, W]
            meta_data: contains any other information regarding the specific video.
                Examples of information can be stored are as follows:
                For SOT, it should countain information such as initial target bbox.
                For long videos, it could contain start and end frame number or timestamp where we
                want to run tracking model.
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
                    },
                    "object_id_2":{},
                    ...
                }
        """
        pass

    @abstractmethod
    def inference_video_handler(
        self, video: av.container.Container, meta_data: Dict = None
    ) -> Dict:
        """
        Run tracking model on a video sequence.

        Args:
            video: a video handler of type av.container.Container to the video file (typically .mp4).
            meta_data: contains any other information regarding the specific video.
                Examples of information can be stored are as follows:
                For SOT, it should countain information such as initial target bbox.
                For long videos, it could contain start and end frame number or timestamp where we
                want to run tracking model.
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
                    },
                    "object_id_2":{},
                    ...
                }
        """
        pass

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
        Typical updates can be: update the local tracking window for SOT, update memory bank of an
        object, and etc.

        Args:
            results: a dictionary holding tracking results of for one image. It is designed to directly
                use the output from self.run_model(image).
        Returns:
            None
        """
        pass

    @abstractmethod
    def init_tracker(self, image: torch.Tensor, meta_data: Dict) -> None:
        """
        This function is used to initilize the tracking model, typically the starting frame.
        This is mostly designed for SOT, but can also be used to initilize MOT.

        Args:
            image: this is one image where we used to initilize the tracker, [1, 3, H, W]
            meta: a dictionary contains information of target objects in this image.
        Returns:
            None
        """
        pass

    @abstractmethod
    def reset_tracker(self) -> None:
        """
        To reset the tracker after done running on one video.
        Clear any intermediate variables to get a clean setup for next video.
        """
        pass
