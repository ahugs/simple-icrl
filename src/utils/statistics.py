from numbers import Number

import numpy as np


class RunningMaxMin:
    """Calculates the running max and min of a data stream.

    :param mean: the initial min estimation for data array. Default to 0.
    :param std: the initial smax estimation for data array. Default to 1.
    :param epsilon: To avoid division by zero.
    """

    def __init__(
        self,
        min: float | np.ndarray = 0.0,
        max: float | np.ndarray = 1.0,
        epsilon: float = np.finfo(np.float32).eps.item(),
    ) -> None:
        self.min, self.max = min, max
        self.eps = epsilon

    def norm(self, data_array: float | np.ndarray) -> float | np.ndarray:
        data_array = (data_array - self.min) / (self.max - self.min)
        return data_array

    def update(self, data_array: np.ndarray) -> None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        batch_min, batch_max = np.min(data_array, axis=0), np.max(data_array, axis=0)
        batch_count = len(data_array)

        self.min = np.min([batch_min, self.min], axis=0)
        self.max = np.max([batch_max, self.max], axis=0)