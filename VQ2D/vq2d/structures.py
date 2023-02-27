from typing import List


class BBox:
    def __init__(self, fno, x1, y1, x2, y2):
        self.fno = fno
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def __repr__(self):
        return "BBox[fno = {}, x1 = {}, y1 = {}, x2 = {}, y2 = {}]".format(
            self.fno, self.x1, self.y1, self.x2, self.y2
        )

    def to_json(self):
        return {
            "fno": int(self.fno),
            "x1": int(self.x1),
            "x2": int(self.x2),
            "y1": int(self.y1),
            "y2": int(self.y2),
        }

    @staticmethod
    def from_json(data):
        return BBox(data["fno"], data["x1"], data["y1"], data["x2"], data["y2"])


class ResponseTrack:
    def __init__(self, bboxes: List[BBox], score: float = None):
        # A set of bounding boxes with time, and an optional confidence score
        self._bboxes = sorted(bboxes, key=lambda x: x.fno)
        if self._check_empty(self._bboxes):
            self._empty_init()
        else:
            self._non_empty_init(self._bboxes)
        self._score = score
        self._check_contiguous()

    @property
    def temporal_extent(self):
        return (self._t_start, self._t_end)

    @property
    def bboxes(self):
        return self._bboxes

    @property
    def length(self):
        return self._length

    @property
    def score(self):
        return self._score

    def has_score(self):
        return self._score is not None

    def _check_empty(self, bboxes):
        return len(bboxes) == 0

    def _empty_init(self):
        self._t_start = 0
        self._t_end = -1
        self._length = 0
        print("Encountered empty track")

    def _non_empty_init(self, bboxes):
        self._t_start = bboxes[0].fno
        self._t_end = bboxes[-1].fno
        self._length = len(bboxes)

    def _check_contiguous(self):
        if self._length != (self._t_end - self._t_start + 1):
            raise ValueError(f"====> ResponseTrack: BBoxes not contiguous")

    def __repr__(self):
        return (
            "ResponseTrack[\n"
            + "\n".join([bbox.__repr__() for bbox in self._bboxes])
            + "]"
        )

    def volume(self):
        v = 0.0
        for bbox in self._bboxes:
            v += bbox.area()
        return v

    def to_json(self):
        score = self._score
        if score is not None:
            score = float(score)
        return {
            "bboxes": [bbox.to_json() for bbox in self._bboxes],
            "score": score,
        }

    @staticmethod
    def from_json(data):
        return ResponseTrack(
            [BBox.from_json(bbox) for bbox in data["bboxes"]],
            data["score"],
        )
