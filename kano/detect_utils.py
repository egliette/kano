from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


def extract_bbox_area(image, bbox):
    """
    Return cropped image from the given bounding box area

    Args:
        image (np.array) with shape (H, W, 3): image to extract the box
        bbox (np.array) with shape (4,): xyxy location of the box

    Returns:
        cropped_image (np.array): with shape (new_H, new_W, 3) based on bbox
    """

    (left, top), (right, bottom) = bbox[:2], bbox[2:]
    return image.copy()[top:bottom, left:right]


def xywh2xyxy(xywh):
    """
    Converts bounding box coordinates from (x_center, y_center, width, height) format to (x_min, y_min, x_max, y_max) format.

    Args:
        xywh (np.array) with shape (4,): A tuple containing (x_center, y_center, width, height) of the bounding box.

    Returns:
        xyxy (tuple(int)): xyxy location of the bounding box.
    """

    x_center, y_center, bbox_width, bbox_height = xywh
    x_min = int(x_center - bbox_width / 2)
    y_min = int(y_center - bbox_height / 2)
    x_max = int(x_center + bbox_width / 2)
    y_max = int(y_center + bbox_height / 2)

    return x_min, y_min, x_max, y_max


def get_font_config(image_height: int) -> Tuple[float, int]:
    """
    Factory function to generate font configuration based on image height.

    Args:
        image_height (int): Height of the image.

    Returns:
        tuple: A tuple containing font scale (float), thickness (int), pad (int).
    """
    if image_height >= 1000:
        return (1.25, 2, 20)
    elif image_height >= 500:
        return (1, 2, 10)
    else:
        return (0.75, 2, 5)


def draw_bbox(
    image: Union[np.ndarray, str],
    bbox: Union[List[float], Tuple[float, ...], np.ndarray],
    bbox_type: str = "xyxy",
    bbox_color: Tuple[int, int, int] = (0, 0, 255),
    label: Optional[str] = None,
) -> np.ndarray:
    """
    Draws a bounding box on the image and optionally draws a multi-line label.

    Args:
        image (np.ndarray or str): The image on which the bounding box will be drawn.
        bbox (list or tuple or np.ndarray): The bounding box coordinates. If it's a list, it should be in the format specified by bbox_type.
        bbox_type (str): Type of bounding box coordinates. Should be either "xyxy" or "xywh" or "s_xywh".
        bbox_color (tuple): Color of the bounding box in BGR format.
        label (str, optional): Label to be displayed alongside the bounding box. Supports multiple lines with '/n' separating lines.

    Returns:
        np.ndarray: Image with the bounding box and label drawn.
    """
    if isinstance(image, str):
        temp_image = cv2.imread(image)
    else:
        temp_image = image.copy()

    image_height, image_width = temp_image.shape[:2]

    if isinstance(bbox, (list, tuple)):
        temp_bbox = np.array(bbox)
    else:
        temp_bbox = bbox.copy()
    if "s_" in bbox_type:
        temp_bbox *= np.array(
            [image_width, image_height, image_width, image_height]
        )

    temp_bbox = temp_bbox.astype(np.int64)

    if "xyxy" in bbox_type:
        x_min, y_min, x_max, y_max = temp_bbox
    elif "xywh" in bbox_type:
        x_min, y_min, x_max, y_max = xywh2xyxy(temp_bbox)
    else:
        raise ValueError("Invalid bounding box type")

    cv2.rectangle(temp_image, (x_min, y_min), (x_max, y_max), bbox_color, 2)

    if label is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness, pad = get_font_config(image_height)

        label_lines = label.split("\n")
        (text_width, text_height), _ = cv2.getTextSize(
            "sample", font, font_scale, thickness
        )
        y_offset = y_min - (text_height + pad) * (len(label_lines) - 1)

        for line in label_lines:
            (text_width, text_height), _ = cv2.getTextSize(
                line, font, font_scale, thickness
            )

            background_position = (x_min, y_offset)
            background_end_position = (
                x_min + text_width,
                y_offset - text_height - pad,
            )
            cv2.rectangle(
                temp_image,
                background_position,
                background_end_position,
                bbox_color,
                -1,
            )
            cv2.putText(
                temp_image,
                line,
                (x_min, y_offset - pad // 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

            y_offset += text_height + pad

    return temp_image


def calculate_iou(xyxy1, xyxy2):
    x1 = max(xyxy1[0], xyxy2[0])
    y1 = max(xyxy1[1], xyxy2[1])
    x2 = min(xyxy1[2], xyxy2[2])
    y2 = min(xyxy1[3], xyxy2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    area_box1 = (xyxy1[2] - xyxy1[0] + 1) * (xyxy1[3] - xyxy1[1] + 1)
    area_box2 = (xyxy2[2] - xyxy2[0] + 1) * (xyxy2[3] - xyxy2[1] + 1)

    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area / union_area

    return iou


class FailedDetectionTypes:
    UnmatchedLabels = 0
    Nothing = 1
    BelowIoUThreshold = 2

    message_dict = {
        0: "Unmatched number of labels",
        1: "Everything is ok",
        2: "IoU between Groundtruth bbox and Predicted bbox is below the given threshold",
    }

    @classmethod
    def print_message(cls, id):
        print(cls.message_dict[id])


def get_failed_detected_results(labels, predictions, iou_threshold=0.5):
    if len(labels) != len(predictions):
        return FailedDetectionTypes.UnmatchedLabels

    checked_prediction_indicies = list()

    for label in labels:
        index = 0
        max_iou = 0
        for i, pred in enumerate(predictions):
            if (
                pred["class"] == label["class"]
                and i not in checked_prediction_indicies
            ):
                iou = calculate_iou(label["xyxy"], pred["xyxy"])
                if iou > max_iou:
                    max_iou = iou
                    index = i

        if index in checked_prediction_indicies:
            return FailedDetectionTypes.UnmatchedLabels
        if max_iou < iou_threshold:
            return FailedDetectionTypes.BelowIoUThreshold

        checked_prediction_indicies.append(index)

    return FailedDetectionTypes.Nothing
