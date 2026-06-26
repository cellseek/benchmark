from abc import ABC, abstractmethod


class DatasetAdapter(ABC):
    #: When True and the model implements ``predict_tracks_returning_masks``, the
    #: benchmark can score segmentation using masks from the tracking pass (no second inference).
    joint_segmentation_with_tracking: bool = False

    @abstractmethod
    def iter_segmentation(self):
        raise NotImplementedError

    @abstractmethod
    def iter_tracking(self):
        raise NotImplementedError
