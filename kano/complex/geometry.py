import numpy as np


def calculate_distance(x1, y1, x2, y2):
    point1 = np.array([x1, y1])
    point2 = np.array([x2, y2])

    distance = np.linalg.norm(point2 - point1)

    return distance
