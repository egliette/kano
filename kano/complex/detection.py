import os

import cv2
import numpy as np

from kano.image_utils import show_image


def extract_bbox_area(image, bbox):
    (left, top), (right, bottom) = bbox
    return image.copy()[top:bottom, left:right]


def xywh2xyxy(xywh):
    x_center, y_center, bbox_width, bbox_height = xywh
    x_min = int(x_center - bbox_width / 2)
    y_min = int(y_center - bbox_height / 2)
    x_max = int(x_center + bbox_width / 2)
    y_max = int(y_center + bbox_height / 2)

    return x_min, y_min, x_max, y_max


def draw_bbox(
    image, bbox, bbox_type="xyxy", bbox_color=(0, 0, 255), label=None
):
    if isinstance(image, str):
        temp_image = cv2.imread(image)
    else:
        temp_image = image.copy()

    image_height, image_width = temp_image.shape[:2]

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
        font_scale = 1
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        background_position = (x_min, y_min)
        background_end_position = (
            x_min + text_width,
            y_min - text_height - 5,
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
            label,
            (x_min, y_min),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

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


class YoloImage:

    def __init__(self, image_path, labels_dict=None):
        self.image = cv2.imread(image_path)
        self.image_path = image_path
        self.label_path = self.get_label_path(image_path)
        self.labels = self.get_labels(self.label_path)
        self.labels_dict = labels_dict

    def get_label_path(self, image_path):
        images_folder_path, image_filename = os.path.split(image_path)
        dataset_path = os.path.dirname(images_folder_path)
        labels_folder_path = os.path.join(dataset_path, "labels")
        label_filename = image_filename[:-4] + ".txt"
        label_path = os.path.join(labels_folder_path, label_filename)

        return label_path

    def get_labels(self, label_path):
        labels = list()
        image_height, image_width = self.image.shape[:2]
        with open(label_path, "r") as file:
            for line in file:
                line = line.strip().split()
                label = {
                    "class": int(line[0]),
                    "s_xywh": np.array([float(x) for x in line[1:]]),
                }
                xywh = label["s_xywh"].copy() * np.array(
                    [image_width, image_height, image_width, image_height]
                )
                label["xyxy"] = xywh2xyxy(xywh)
                labels.append(label)
        return labels

    def show_image(self, figsize=(10, 10)):
        show_image(self.image, figsize)

    def get_annotated_image(self):
        annotated_image = self.image.copy()
        for i, label in enumerate(self.labels):
            cls = label["class"]
            if self.labels_dict is not None:
                cls = self.labels_dict[cls]
            bbox = label["s_xywh"]
            annotated_image = draw_bbox(
                annotated_image, bbox, "s_xywh", (0, 255, 0), str(cls)
            )

        return annotated_image

    def show_annotated_image(self, figsize=(10, 10)):
        annotated_image = self.get_annotated_image()
        show_image(annotated_image, figsize)
