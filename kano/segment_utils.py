import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_bounding_box_of_mask(mask):
    """
    Input:
        mask (2D numpy array): the value of mask is 0 and 1
    Output:
        top_left_point (numpy array with shape = (2,)): coordinate of bbox
        bottom_right_point (numpy array with shape = (2,)): coordinate of bbox
    """
    nonzero_points = cv2.findNonZero(mask)
    x, y, width, height = cv2.boundingRect(nonzero_points)

    return (np.array([x, y]), np.array([x + width, y + height]))


def show_mask(mask, figsize=None):
    if figsize:
        plt.figure(figsize=figsize)
    plt.imshow(mask, cmap="gray")
    plt.show()
