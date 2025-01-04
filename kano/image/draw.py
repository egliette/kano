from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class Location(Enum):
    TopLeft = 1
    TopRight = 2
    BottomLeft = 3
    BottomRight = 4


def calculate_text_coordinates(
    point: Tuple[int, int],
    location: Location,
    max_width: int,
    total_height: int,
    line_heights: List[int],
    pad: int,
) -> Dict[str, Tuple[int, int]]:
    """Calculate coordinates for text and background box with padding"""
    if location == Location.TopLeft:
        box_start = (point[0] - pad, point[1] - pad)
        box_end = (point[0] + max_width + pad, point[1] + total_height + pad)
        x_text = point[0]
        y_text = point[1] + line_heights[0]
    elif location == Location.TopRight:
        box_start = (point[0] - max_width - pad, point[1] - pad)
        box_end = (point[0] + pad, point[1] + total_height + pad)
        x_text = point[0] - max_width
        y_text = point[1] + line_heights[0]
    elif location == Location.BottomLeft:
        box_start = (point[0] - pad, point[1] - total_height - pad)
        box_end = (point[0] + max_width + pad, point[1] + pad)
        x_text = point[0]
        y_text = point[1] - total_height + line_heights[0]
    else:  # BottomRight
        box_start = (point[0] - max_width - pad, point[1] - total_height - pad)
        box_end = (point[0] + pad, point[1] + pad)
        x_text = point[0] - max_width
        y_text = point[1] - total_height + line_heights[0]

    return {
        "box_start": box_start,
        "box_end": box_end,
        "x_text": x_text,
        "y_text": y_text,
    }


def add_text(
    image: np.ndarray,
    text: str,
    point: Tuple[int, int],
    location: Location = Location.TopLeft,
    font_scale: float = 2,
    font_thickness: int = 3,
    pad: int = 20,
    font_color: Tuple[int, int, int] = (0, 255, 0),
    background_color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Add multiline text to an image at specified location with optional background box.

    Args:
        image: Input image as numpy array
        text: Text to be added (can contain \n for new lines)
        point: Starting point coordinates (x, y)
        location: Position enum indicating text anchor point
        font_scale: Scale of the font
        font_thickness: Thickness of the font
        font_color: BGR color tuple for the font
        background_color: BGR color tuple for the background box, None for no background
        pad: Padding around text inside background box

    Returns:
        Modified image as numpy array
    """
    img_copy = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = text.split("\n")

    # Calculate size for each line
    line_sizes = [
        cv2.getTextSize(line, font, font_scale, font_thickness)
        for line in lines
    ]

    # Get maximum width and total height
    max_width = max(size[0][0] for size in line_sizes)
    total_height = sum(size[0][1] + size[1] for size in line_sizes)
    line_heights = [size[0][1] + size[1] for size in line_sizes]

    # Add padding between lines
    line_spacing = line_heights[0] // 2 if line_heights else 0
    total_height += line_spacing * (len(lines) - 1)

    # Get coordinates from factory function
    coords = calculate_text_coordinates(
        point, location, max_width, total_height, line_heights, pad
    )

    # Draw background if specified
    if background_color is not None:
        cv2.rectangle(
            img_copy,
            coords["box_start"],
            coords["box_end"],
            background_color,
            -1,
        )

    # Draw each line
    top_pad = line_spacing // 3
    current_y = coords["y_text"]
    for i, (line, line_height) in enumerate(zip(lines, line_heights)):
        if i > 0:
            current_y += line_height + line_spacing
        text_org = (coords["x_text"], current_y - top_pad)
        cv2.putText(
            img_copy,
            line,
            text_org,
            font,
            font_scale,
            font_color,
            font_thickness,
        )

    return img_copy
