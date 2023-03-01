import torch
import torch.nn.functional as F
import tracking.dataset.processing.processing_utils as prutils
from tracking.dataset.processing.base_processing import BaseProcessing
from tracking.utils.tensor import TensorDict


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class STARKProcessing(BaseProcessing):
    """The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(
        self,
        search_area_factor,
        output_sz,
        center_jitter_factor,
        scale_jitter_factor,
        mode="pair",
        *args,
        **kwargs,
    ):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        """Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(
            torch.randn(2) * self.scale_jitter_factor[mode]
        )
        max_offset = (
            jittered_size.prod().sqrt()
            * torch.tensor(self.center_jitter_factor[mode]).float()
        )
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform["joint"] is not None:
            (
                data["template_images"],
                data["template_anno"],
                data["template_masks"],
            ) = self.transform["joint"](
                image=data["template_images"],
                bbox=data["template_anno"],
                mask=data["template_masks"],
            )
            (
                data["search_images"],
                data["search_anno"],
                data["search_masks"],
            ) = self.transform["joint"](
                image=data["search_images"],
                bbox=data["search_anno"],
                mask=data["search_masks"],
                new_roll=False,
            )

        for s in ["template", "search"]:
            assert (
                self.mode == "sequence" or len(data[s + "_images"]) == 1
            ), "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + "_anno"]]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = (
                torch.stack(jittered_anno, dim=0)[:, 2],
                torch.stack(jittered_anno, dim=0)[:, 3],
            )

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data["valid"] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(
                data[s + "_images"],
                jittered_anno,
                data[s + "_anno"],
                self.search_area_factor[s],
                self.output_sz[s],
                masks=data[s + "_masks"],
            )
            # Apply transforms
            (
                data[s + "_images"],
                data[s + "_anno"],
                data[s + "_att"],
                data[s + "_masks"],
            ) = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False
            )

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + "_att"]:
                if (ele == 1).all():
                    data["valid"] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + "_att"]:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(
                    torch.bool
                )[0]
                if (mask_down == 1).all():
                    data["valid"] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data["valid"] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros(
                (1, self.output_sz["template"], self.output_sz["template"])
            )
            data["search_masks"] = torch.zeros(
                (1, self.output_sz["search"], self.output_sz["search"])
            )
        # Prepare output
        if self.mode == "sequence":
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
