import copy
from typing import List, Optional

import cv2
import numpy as np


def rotate_image(
    image: np.ndarray, degree: float, expand: bool = False
) -> np.ndarray:
    """
    Rotate the given image with given degree

    Args:
        image (np.ndarray): numpy array of the image
        degree (float): total degrees to rotate
        expand (bool): Pad rotated image with black color if True,
            otherwise return the cropped rotated image

    Returns:
        rotated_image (np.ndarray): numpy array of the rotated image
    """
    height, width = image.shape[:2]
    center_point = (width / 2, height / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center_point, degree, 1.0)

    new_height, new_width = image.shape[:2]
    if expand:
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])

        new_width = int(height * abs_sin + width * abs_cos)
        new_height = int(height * abs_cos + width * abs_sin)

        rotation_matrix[0, 2] += new_width / 2 - center_point[0]
        rotation_matrix[1, 2] += new_height / 2 - center_point[1]

    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (new_width, new_height)
    )

    return rotated_image


def resize_with_black_padding(
    image: np.ndarray, new_height: int, new_width: int
) -> np.ndarray:
    """Resize the image with black padding to fit new dimensions."""
    original_height, original_width = image.shape[:2]

    if len(image.shape) == 3:
        padded_image = np.zeros(
            (new_height, new_width, image.shape[2]), dtype=np.uint8
        )
        padded_image[:original_height, :original_width, :] = image
    else:
        padded_image = np.zeros((new_height, new_width), dtype=np.uint8)
        padded_image[:original_height, :original_width] = image

    return padded_image


def shift_image(
    image: np.ndarray,
    dx: int,
    dy: int,
    new_height: Optional[int] = None,
    new_width: Optional[int] = None,
) -> np.ndarray:
    """Shift the image by dx and dy, with optional resizing."""
    padded_image = image.copy()

    if new_height and new_width:
        padded_image = resize_with_black_padding(
            padded_image, new_height, new_width
        )

    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_image = cv2.warpAffine(
        padded_image,
        translation_matrix,
        (padded_image.shape[1], padded_image.shape[0]),
    )
    return shifted_image


def pad_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """Pad the image to the target size."""
    height, width = image.shape[:2]

    pad_height = max(0, target_size[0] - height)
    pad_width = max(0, target_size[1] - width)

    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad

    padded_image = cv2.copyMakeBorder(
        image.copy(),
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    return padded_image


def concatenate_images(
    image_list: List[List[np.ndarray]], padding_size: int = 0
) -> np.ndarray:
    """
    Concatenate images based on its appearance order in the given 2D list
    Each image will be padded to the max height, max width in the given list.

    Args:
        image_list (list(list(np.ndarray))): 2D (or 1D) list of images
        padding_size (int): padding distance between each image

    Returns:
        concatenated_image (np.ndarray): the concatenated image from 2D list
    """

    # ensure image_list is a 2D list
    new_image_list = copy.deepcopy(image_list)
    if not isinstance(image_list[0], list):
        new_image_list = [copy.deepcopy(image_list)]

    rows = len(new_image_list)
    cols = max(len(row) for row in new_image_list)

    for image_row in new_image_list:
        image_row += [None] * (cols - len(image_row))

    # pad every images
    max_height = max(
        image.shape[0]
        for row in new_image_list
        for image in row
        if image is not None
    )
    max_width = max(
        image.shape[1]
        for row in new_image_list
        for image in row
        if image is not None
    )

    padded_image_list = list()
    for i, row in enumerate(new_image_list):
        padded_image_list.append(list())
        for j, image in enumerate(row):
            padded_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            if image is not None:
                padded_image = pad_image(image, (max_height, max_width))
            padded_image_list[i].append(padded_image)

    concatenated_image = np.zeros(
        (
            max_height * rows + padding_size * (rows - 1),
            max_width * cols + padding_size * (cols - 1),
            3,
        ),
        dtype=np.uint8,
    )

    for i, row in enumerate(padded_image_list):
        for j, image in enumerate(row):

            h, w, _ = image.shape
            concatenated_image[
                (i > 0) * padding_size
                + i * max_height : (i > 0) * padding_size
                + i * max_height
                + h,
                (j > 0) * padding_size
                + j * max_width : (j > 0) * padding_size
                + j * max_width
                + w,
            ] = image

    return concatenated_image
