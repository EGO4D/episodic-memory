import torchvision.transforms as transforms
from tracking.utils.tensor import TensorDict


class BaseProcessing:
    """Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
    through the network. For example, it can be used to crop a search region around the object, apply various data
    augmentations, etc."""

    def __init__(
        self,
        transform=transforms.ToTensor(),
        template_transform=None,
        search_transform=None,
        joint_transform=None,
    ):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {
            "template": transform if template_transform is None else template_transform,
            "search": transform if search_transform is None else search_transform,
            "joint": joint_transform,
        }

    def __call__(self, data: TensorDict):
        raise NotImplementedError
