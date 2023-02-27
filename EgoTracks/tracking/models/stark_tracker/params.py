from typing import Dict

from tracking.models.stark_tracker.config.stark_st2.config import cfg


class TrackerParams:
    """Class for tracker parameters."""

    def set_default_values(self, default_vals: Dict):
        for name, val in default_vals.items():
            if not hasattr(self, name):
                setattr(self, name, val)

    def get(self, name: str, *default):
        """Get a parameter value with the given name. If it does not exists, it return the default value given as a
        second argument or returns an error if no default value is given."""
        if len(default) > 1:
            raise ValueError("Can only give one default value.")

        if not default:
            return getattr(self, name)

        return getattr(self, name, default[0])

    def has(self, name: str):
        """Check if there exist a parameter with the given name."""
        return hasattr(self, name)


def parameters(yaml_name: str):
    params = TrackerParams()
    # prj_dir = ""
    # save_dir = ""
    # update default config from yaml file
    # yaml_file = os.path.join(prj_dir, 'experiments/stark_st2/%s.yaml' % yaml_name)
    # update_config_from_file(yaml_file)
    params.cfg = cfg
    # print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = "manifold://tracking/tree/models/STARK/STARKST_ep0050.pth"

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
