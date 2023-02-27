#!/usr/bin/env python3

"""
Object registration classes and helper functions. The following classes
provide functionality to register instances given images. Those images
will can be used as template images to track objects.

Intake from
vision/fair_accel/pixar_env/pixar_environment/visual_query/visual_query/utils/register_helper.py
"""

from typing import Any, Iterator, List


class TemplateImage:
    """
    Define a template image.

    The visual query pipeline will use at least one
    template image of a given object instance to look for that object in the video.
    """

    def __init__(
        self,
        image: Any,
        bbox: List = None,
        features: Any = None,
        cropped: bool = True,
    ) -> None:
        """
        Initialize a TemplateImage object.

        Args:
            image: the image to be used as a template
            bbox (list): the bounding box for the object in the image (will be None if the object
                region is cropped, aka cropped=True)
            features: the feature map for the image
            cropped (bool): will be True if the object region is already cropped
        """
        self.image = image
        self.bbox = bbox
        self.features = features
        self.cropped = cropped

    def update_template(self, attribute: str, value: Any) -> None:
        """
        Update an attribute of the TemplateImage object with the given value.

        For example, `update_template("features", value)` means `self.features = value`

        Args:
            attribute (str): the string of the attribute to be updated
            value (any): the new value of the desired attribute to change

        Returns:
            None
        """
        if attribute == "image":
            self.image = value
        elif attribute == "bbox":
            self.bbox = value
        elif attribute == "features":
            self.features = value
        elif attribute == "cropped":
            self.cropped = value


class ObjectInstance:
    """
    An object instance with a list of TemplateImage template images.

    The ObjectInstance class organizes all template images for a given instance.
    """

    def __init__(self, description: str = "") -> None:
        """
        Initialize a new object instance.
        """
        self.description = description
        self.templates = []

    def add_template(
        self, image: Any, bbox: List = None, features: Any = None, cropped: bool = True
    ) -> None:
        """
        Add a new object template to the list of templates.

        Args:
            image: the image to be used as a template
            bbox: the bounding box for the object in the image (will be None if the object
                region is cropped, aka cropped=True)
            features: the feature map for the image
            cropped (bool): will be True if the object region is already cropped
        """
        # instantiate TemplateImage to add a new template image to the list
        template = TemplateImage(image, bbox, features, cropped)
        self.templates.append(template)

    def get_templates_list(self) -> List:
        """
        Returns the list of templates.

        Args:
            None

        Returns:
            self.templates (list): the list of TemplateImage templates
        """
        return self.templates

    def __iter__(self):
        """
        Gets an iterator object through the templates list.

        Returns:
            iterator: an iterator for the templates list
        """
        return iter(self.templates)

    def __len__(self) -> int:
        """
        Gets the number of templates

        Args:
            None

        Returns:
            int: the length of the templates list
        """
        return len(self.templates)


class InstanceRegistration:
    """
    Define the InstanceRegistration class to hold all registered instances.

    The instance register will esentially be a dictionary containing all object
    instances corresponding to their instance ID's.
    The structure will be:
    instances = {
        "instance_id_1": ObjectInstance(),
        "instance_id_2": ObjectInstance(),
    }
    """

    def __init__(self) -> None:
        """
        Initialize the InstanceRegistration class.
        """
        self.instances = {}  # the dictionary to hold all registered instances

    def add(
        self,
        instance_id: str,
        image: Any,
        bbox: List = None,
        path: str = None,
        features: Any = None,
        cropped: bool = True,
        description: str = None,
    ) -> None:
        """
        Register a new instance and add it to the InstanceRegistration object.

        If the instance ID is not in the register, add a new ID and ObjectInstance
        pair to the dictionary. If instance ID already exists, append new templates
        to the ObjectInstance value.

        Args:
            instance_id (str): the instance ID number
            image: the image to be used as a template
            bbox: the bounding box for the object in the image (will be None if the object
                region is cropped, aka cropped=True)
            features: the feature map for the image
            cropped (bool): will be True if the object region is already cropped
            description (str): a string description of the object instance
        """
        if instance_id not in self.instances:
            self.instances[instance_id] = ObjectInstance(description)

        self.instances[instance_id].add_template(image, bbox, features, cropped)

    def remove(self, instance_id: str) -> None:
        """
        Remove an instance from the register.

        Args:
            instance_id (str): the instance id (number) to remove

        Returns:
            None
        """
        if instance_id in self.instances:
            self.instances.pop(instance_id)

    def get_instance(self, instance_id: str) -> ObjectInstance:
        """
        Get the instance associated with a given instance id.

        Args:
            instance_id (str): the instance id (number)

        Returns:
            ObjectInstance: an instance from the instances dictionary of type ObjectInstance
        """
        return self.instances[instance_id]

    def get_instance_ids(self) -> List:
        """
        Gets the instance ids from the instances dictionary.

        Args:
            None

        Returns:
            keys (list): a list of keys (instance ids) from the instances dictionary
        """
        return list(self.instances.keys())

    def __iter__(self) -> Iterator:
        """
        Get an iterator over the instances dictionary.

        Returns an iterator where each element is a tupe of (key, value)
        pairs from the instances dictionary. Keys are instance ids, values are
        instances of type ObjectInstance

        Args:
            None

        Returns:
            iterator: an iterator through the items of the dictionary
        """
        return iter(self.instances.items())

    def __len__(self) -> int:
        """
        Gets the number of instances in the instance dictionary.

        Args:
            None

        Returns:
            int: the length of the instance dictionary
        """
        return len(self.instances)
