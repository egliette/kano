import numpy as np

from kano.lab.box_gen.box import Box


def solve_equation(
    elapsed_time: float,
    duration: float,
    first_val: np.ndarray,
    last_val: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolates between two values based on elapsed time and duration.

    Args:
        elapsed_time (float): The elapsed time since the start of the transition.
        duration (float): The total duration of the transition.
        first_val (np.ndarray): The initial value.
        last_val (np.ndarray): The final value.

    Returns:
        np.ndarray: The interpolated value.
    """
    return elapsed_time / duration * (last_val - first_val) + first_val


class FakeDetect:
    """
    Simulates an animated transition between two boxes over a specified duration.
    """

    def __init__(
        self,
        begin_box: Box,
        end_box: Box,
        duration: float,
        start_after: float = 0,
        from_bottom: bool = False,
    ) -> None:
        """
        Initializes the FakeDetect instance.

        Args:
            begin_box (Box): The starting box of the transition.
            end_box (Box): The ending box of the transition.
            duration (float): The duration of the transition in seconds.
            start_after (float): The delay before starting the transition. Default is 0.
            from_bottom (bool): Whether to use bottom-aligned coordinates. Default is False.
        """
        self.begin_box = begin_box
        self.end_box = end_box
        self.duration = duration
        self.start_after = start_after
        self.from_bottom = from_bottom
        self.start_time: float = None

    def reverse(self) -> None:
        """
        Reverses the transition by swapping the starting and ending boxes.
        """
        self.begin_box, self.end_box = self.end_box, self.begin_box

    def start(self, current_time: float) -> None:
        """
        Starts the transition at the specified current time.

        Args:
            current_time (float): The current time to start the transition.
        """
        self.start_time = current_time

    def _get_current_box(self, elapsed_time: float) -> Box:
        """
        Computes the intermediate box based on the elapsed time.

        Args:
            elapsed_time (float): The elapsed time since the transition started.

        Returns:
            Box: The interpolated box at the current state of the transition.
        """
        new_width = solve_equation(
            elapsed_time,
            self.duration,
            self.begin_box.width,
            self.end_box.width,
        )
        new_height = solve_equation(
            elapsed_time,
            self.duration,
            self.begin_box.height,
            self.end_box.height,
        )
        new_point = solve_equation(
            elapsed_time,
            self.duration,
            self.begin_box.point,
            self.end_box.point,
        )
        return Box(
            width=int(new_width),
            height=int(new_height),
            point=new_point,
        )

    def move(self, current_time: float) -> np.ndarray:
        """
        Computes the current box's coordinates in (xmin, ymin, xmax, ymax) format.

        Args:
            current_time (float): The current time used to compute the box's state.

        Returns:
            np.ndarray: The current box's coordinates, or None if the transition has not started or is finished.

        Raises:
            ValueError: If the start time is not set.
        """
        if self.start_time is None:
            raise ValueError(
                "Start time is not set. Call start() method first."
            )

        elapsed_time = current_time - self.start_time
        if (
            elapsed_time < self.start_after
            or elapsed_time > self.duration + self.start_after
        ):
            return None

        return self._get_current_box(elapsed_time).get_xyxy(self.from_bottom)

    def is_end(self, current_time: float) -> bool:
        """
        Determines whether the transition has finished.

        Args:
            current_time (float): The current time to check against the transition's end time.

        Returns:
            bool: True if the transition has ended, False otherwise.

        Raises:
            ValueError: If the start time is not set.
        """
        if self.start_time is None:
            raise ValueError(
                "Start time is not set. Call start() method first."
            )

        elapsed_time = current_time - self.start_time
        return elapsed_time > self.duration + self.start_after
