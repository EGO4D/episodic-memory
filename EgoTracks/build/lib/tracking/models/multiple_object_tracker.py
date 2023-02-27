from abc import abstractmethod

from tracking.models.tracker import Tracker


class MultipleObjectTracker(Tracker):
    """
    Define a base class for multiple object tracker (MOT).

    TODO: we have not looked much into MOT. Create a placeholder for future MOT models.
    """

    def __init__(self, model):
        super().__init__(model)

    @abstractmethod
    def inference(self, video, meta_data):
        pass

    @abstractmethod
    def run_model(self, img):
        pass

    @abstractmethod
    def update_tracker(self, res):
        pass

    @abstractmethod
    def init_tracker(self, img, meta):
        pass
