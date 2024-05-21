import cv2
import numpy as np


def draw_skeleton(image, keypoints):
    if isinstance(image, str):
        full_image = cv2.imread(image)
    else:
        full_image = image.copy()

    # draw points
    head_indices = [0, 1, 2, 3, 4]
    body_indices = [5, 6, 7, 8, 9, 10]
    # leg_indices = [11, 12, 13, 14, 15, 16]

    for i, keypoint in enumerate(keypoints):
        if i in head_indices:
            color = (255, 0, 0)
        elif i in body_indices:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        xy = keypoint["xy"]
        state = keypoint["state"]

        if state == 1:
            thickness = 2
        else:
            thickness = -1

        if xy != [0, 0] and state != 0:
            cv2.circle(
                full_image, xy, radius=10, color=color, thickness=thickness
            )

    # draw lines
    head_lines = [6, 4, 2, 0, 1, 3, 5]
    body_lines = [10, 8, 6, 5, 7, 9]
    leg_lines = [16, 14, 12, 11, 13, 15]

    line_thickness = 2

    for id1, id2 in zip(head_lines[:-1], head_lines[1:]):
        if keypoints[id1]["state"] != 0 and keypoints[id2]["state"] != 0:
            cv2.line(
                full_image,
                keypoints[id1]["xy"],
                keypoints[id2]["xy"],
                (255, 0, 0),
                line_thickness,
            )

    for id1, id2 in zip(body_lines[:-1], body_lines[1:]):
        if keypoints[id1]["state"] != 0 and keypoints[id2]["state"] != 0:
            cv2.line(
                full_image,
                keypoints[id1]["xy"],
                keypoints[id2]["xy"],
                (0, 0, 255),
                line_thickness,
            )

    if keypoints[12]["state"] != 0 and keypoints[6]["state"] != 0:
        cv2.line(
            full_image,
            keypoints[12]["xy"],
            keypoints[6]["xy"],
            (0, 0, 255),
            line_thickness,
        )
    if keypoints[11]["state"] != 0 and keypoints[5]["state"] != 0:
        cv2.line(
            full_image,
            keypoints[11]["xy"],
            keypoints[5]["xy"],
            (0, 0, 255),
            line_thickness,
        )

    for id1, id2 in zip(leg_lines[:-1], leg_lines[1:]):
        if keypoints[id1]["state"] != 0 and keypoints[id2]["state"] != 0:
            cv2.line(
                full_image,
                keypoints[id1]["xy"],
                keypoints[id2]["xy"],
                (0, 255, 0),
                line_thickness,
            )

    return full_image
