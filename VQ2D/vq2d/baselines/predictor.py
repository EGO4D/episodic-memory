from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from detectron2.engine import DefaultPredictor


class SiamPredictor(DefaultPredictor):
    def __call__(
        self,
        original_images: Sequence[np.ndarray],
        visual_crops: Sequence[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Args:
            original_images (np.ndarray): a list of images of shape (H, W, C) (in BGR order).
            visual_crops (np.ndarray): a list of images of shape (H, W, C) (in BGR order)

        Returns:
            predictions (list[dict]):
                the output of the model for a list of images.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            inputs = []
            for original_image, visual_crop in zip(original_images, visual_crops):
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                    visual_crop = visual_crop[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(
                    original_image
                )
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                reference = torch.as_tensor(
                    visual_crop.astype("float32").transpose(2, 0, 1)
                )
                inputs.append(
                    {
                        "image": image,
                        "height": height,
                        "width": width,
                        "reference": reference,
                    }
                )
            predictions = self.model(inputs)
            return predictions
