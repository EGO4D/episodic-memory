from .kys import KYSRunner
from .particle_filter import PFRunner


class Tracker:
    """
    A generic tracker that selects a specific version based on config.
    The set of existing trackers implemented. This can be extended to
    include more complex ones.
    """

    valid_trackers = ["pfilter", "kys"]

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        assert self.cfg.tracker.type in self.valid_trackers, (
            f"Tracker type {self.cfg.tracker.type} is invalid!"
            f"Please select one of the following: {self.valid_trackers}"
        )
        self.initialize_runner()

    def initialize_runner(self):
        if self.cfg.tracker.type == "pfilter":
            self.tracker = PFRunner(self.cfg, self.device)
        elif self.cfg.tracker.type == "kys":
            self.tracker = KYSRunner(self.cfg, self.device)

    def __call__(self, *args, **kwargs):
        outputs = self.tracker(*args, **kwargs)
        return outputs
