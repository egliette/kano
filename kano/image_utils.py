from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests


def show_image(image, figsize=None):

    if isinstance(image, str):
        temp_image = cv2.imread(image)
    else:
        temp_image = image.copy()

    if figsize:
        plt.figure(figsize=figsize)
    temp_image = cv2.cvtColor(temp_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    plt.imshow(temp_image)
    plt.show()


def download_image(url, save_path=None):
    response = requests.get(url)
    if response.status_code == 200:
        image_stream = BytesIO(response.content)
        image_data = np.frombuffer(image_stream.read(), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if save_path:
            cv2.imwrite(save_path, image)

        return image
    else:
        print(f"Failed to download image from {url}")
        return None


def get_random_picture(width=400, height=300, save_path=None):
    image = download_image(
        f"https://picsum.photos/{width}/{height}", save_path
    )
    return image


def rotate_image(image, degree, expand=False):
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


def resize_with_black_padding(image, new_height, new_width):
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


def shift_image(image, dx, dy, new_height=None, new_width=None):
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
