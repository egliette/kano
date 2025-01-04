from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class Box:
    width: int
    height: int
    point: Union[np.ndarray, list, tuple]

    def __post_init__(self) -> None:
        """
        Ensures the point is a 2D numpy array. Converts from a list or tuple if necessary.
        Raises:
            ValueError: If the point is not a 2D numpy array or does not have exactly two elements.
        """
        if isinstance(self.point, (list, tuple)):
            self.point = np.array(self.point)
        if not (
            isinstance(self.point, np.ndarray) and self.point.shape == (2,)
        ):
            raise ValueError(
                "Point must be a numpy array or a list/tuple with exactly two elements."
            )

    def get_xyxy(self, from_bottom: bool = False) -> np.ndarray:
        """
        Returns the box coordinates in (xmin, ymin, xmax, ymax) format.
        Args:
            from_bottom (bool): Whether to use the point as a bottom point (feet) to calculate box coordinate.
        Returns:
            np.ndarray: Coordinates in the format [xmin, ymin, xmax, ymax].
        """
        cx, cy = self.point
        if from_bottom:
            return np.array(
                [
                    cx - self.width / 2,
                    cy,
                    cx + self.width / 2,
                    cy + self.height,
                ],
                dtype=int,
            )
        return np.array(
            [
                cx - self.width / 2,
                cy - self.height / 2,
                cx + self.width / 2,
                cy + self.height / 2,
            ],
            dtype=int,
        )
