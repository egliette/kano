from enum import Enum
from typing import List

from kano.lab.box_gen.box import Box
from kano.lab.box_gen.fake_detect import FakeDetect


class InvalidEnumValueError(ValueError):
    """
    Custom exception for invalid enum values.
    """

    def __init__(self, enum_class: Enum, invalid_value: object) -> None:
        """
        Initializes the error with details about the invalid value and valid options.

        Args:
            enum_class (Enum): The enum class being validated.
            invalid_value (object): The invalid value passed.

        Raises:
            ValueError: Describes the invalid value and valid options.
        """
        valid_values = ", ".join([e.name for e in enum_class])
        self.message = f"Invalid enum value '{invalid_value}', expected one of: {valid_values}"
        super().__init__(self.message)


class LoopType(Enum):
    """
    Enumeration representing loop types for the detection sequence.
    """

    NoLoop = 1
    Reversed = 2
    Replay = 3


class DetectGen:
    """
    Class to manage and generate a sequence of fake detections for animation or simulation purposes.
    """

    def __init__(
        self,
        boxes: List[Box],
        durations: List[float],
        start_after: float = 0,
        from_bottom: bool = False,
        loop_type: LoopType = LoopType.NoLoop,
    ) -> None:
        """
        Initializes the DetectGen object with the given boxes and durations.

        Args:
            boxes (List[Box]): A list of Box objects representing the sequence.
            durations (List[float]): Durations for transitions between boxes.
            start_after (float): Initial delay before starting the sequence. Default is 0.
            from_bottom (bool): Whether to use bottom-aligned coordinates. Default is False.
            loop_type (LoopType): Type of looping for the sequence. Default is NoLoop.

        Raises:
            InvalidEnumValueError: If the loop_type is not a valid LoopType.
        """
        if loop_type not in LoopType.__members__.values():
            raise InvalidEnumValueError(LoopType, loop_type)

        self.loop_type = loop_type
        self.fake_detects = self.get_sequence_detect(
            boxes, durations, start_after, from_bottom
        )
        self.start_time: float = None
        self.curr_i: int = 0
        self.stop: bool = False

    def start(self, current_time: float) -> None:
        """
        Starts the sequence at the specified current time.

        Args:
            current_time (float): The current time to start the sequence.
        """
        self.start_time = current_time
        self.fake_detects[0].start(current_time)

    def _update_i(self, current_time: float) -> None:
        """
        Updates the current index of the detection sequence based on the current time.

        Args:
            current_time (float): The current time used to update the sequence.
        """
        if not self.fake_detects[self.curr_i].is_end(current_time):
            return

        if self.curr_i < len(self.fake_detects) - 1:
            self.start_time += self.fake_detects[self.curr_i].duration
            self.curr_i += 1
            self.fake_detects[self.curr_i].start(self.start_time)
            return

        if self.loop_type == LoopType.NoLoop:
            self.stop = True
            return

        if self.loop_type == LoopType.Reversed:
            self.reverse_sequence()
            return

        if self.loop_type == LoopType.Replay:
            self.replay_sequence()
            return

    def reverse_sequence(self) -> None:
        """
        Reverses the sequence of detections.
        """
        self.fake_detects[0].start_after = 0
        self.start_time += self.fake_detects[self.curr_i].duration
        for fake_det in self.fake_detects:
            fake_det.reverse()
        self.fake_detects = self.fake_detects[::-1]
        self.curr_i = 0
        self.fake_detects[self.curr_i].start(self.start_time)

    def replay_sequence(self) -> None:
        """
        Replays the detection sequence from the beginning.
        """
        self.fake_detects[0].start_after = 0
        self.start_time += self.fake_detects[self.curr_i].duration
        self.curr_i = 0
        self.fake_detects[self.curr_i].start(self.start_time)

    def gen_xyxy(self, current_time: float) -> bool:
        """
        Generates the current (xmin, ymin, xmax, ymax) coordinates for the sequence.

        Args:
            current_time (float): The current time for coordinate generation.

        Returns:
            bool: True if the sequence has ended, otherwise False.
        """
        self._update_i(current_time)
        if self.stop:
            return True
        return self.fake_detects[self.curr_i].move(current_time)

    def get_sequence_detect(
        self,
        boxes: List[Box],
        durations: List[float],
        start_after: float = 0,
        from_bottom: bool = False,
    ) -> List[FakeDetect]:
        """
        Creates a sequence of FakeDetect objects based on the provided boxes and durations.

        Args:
            boxes (List[Box]): A list of Box objects representing the sequence.
            durations (List[float]): A list of durations for transitions between boxes.
            start_after (float): Initial delay before starting the sequence. Default is 0.
            from_bottom (bool): Whether to use bottom-aligned coordinates. Default is False.

        Returns:
            List[FakeDetect]: A list of FakeDetect objects for the sequence.

        Raises:
            ValueError: If there are fewer than 2 boxes or if the number of durations is invalid.
        """
        if len(boxes) < 2:
            raise ValueError("Need at least 2 boxes")

        if len(boxes) != len(durations) + 1:
            raise ValueError(
                f"The number of boxes must be exactly one more than the number of durations. "
                f"Got {len(boxes)} boxes and {len(durations)} durations."
            )

        fake_detects = []
        for i in range(len(boxes) - 1):
            fake_detects.append(
                FakeDetect(
                    begin_box=boxes[i],
                    end_box=boxes[i + 1],
                    duration=durations[i],
                    start_after=start_after,
                    from_bottom=from_bottom,
                )
            )
        return fake_detects
