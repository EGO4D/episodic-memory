from typing import Dict, Tuple, Any

import torch
from detectron2.config import CfgNode, configurable
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage

__all__ = ["SiameseRCNN"]


@META_ARCH_REGISTRY.register()
class SiameseRCNN(GeneralizedRCNN):
    """
    Siamese R-CNN. Any model that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    In addition to GeneralizedRCNN, this takes in a reference crop as input,
    extracts features from it, and passes it to the ROIHead.
    """

    @configurable
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg: CfgNode) -> Dict[str, Any]:
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(
                cfg, backbone.output_shape()
            ),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * reference: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "scores"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images, references = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        ref_features = self.backbone(references.tensor)
        ref_features = {k: v.view(-1, 1, *v.shape[1:]) for k, v in ref_features.items()}

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(
            images, features, proposals, ref_features, gt_instances
        )
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        do_postprocess: bool = True,
    ) -> Any:
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images, references = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        ref_features = self.backbone(references.tensor)
        ref_features = {k: v.view(-1, 1, *v.shape[1:]) for k, v in ref_features.items()}

        if self.proposal_generator is not None:
            proposals, _ = self.proposal_generator(images, features, None)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        results, _ = self.roi_heads(images, features, proposals, ref_features, None)

        if do_postprocess:
            assert (
                not torch.jit.is_scripting()
            ), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(
                results, batched_inputs, images.image_sizes
            )
        else:
            return results

    def preprocess_image(
        self, batched_inputs: Tuple[Dict[str, torch.Tensor]]
    ) -> Tuple[ImageList, ImageList]:
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        references = [x["reference"].to(self.device) for x in batched_inputs]
        references = [(x - self.pixel_mean) / self.pixel_std for x in references]
        references = ImageList.from_tensors(references, 0)
        return images, references
