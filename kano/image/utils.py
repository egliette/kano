from io import BytesIO
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests


def show_image(image, figsize=(10, 10)):
    """
    Show image from a numpy array or a file path.
    Which can be used when run .py or .ipynb files.

    Args:
        image (Union[np.ndarray, str]): a numpy array or a file path
        figsize (Tuple[int, int]): (width, height) for image to show
    """
    if isinstance(image, str):
        temp_image = cv2.imread(image)
    else:
        temp_image = image.copy()

    plt.figure(figsize=figsize)
    temp_image = cv2.cvtColor(temp_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    plt.imshow(temp_image)
    plt.show()


def save_image(image: np.ndarray, save_path: str) -> None:
    """Save the image to a file."""
    cv2.imwrite(save_path, image)


def download_image(url, save_path=None):
    """
    Download image from given url

    Args:
        url (str): url of the image
        save_path (str): path to save image

    Returns:
        image (Union[np.ndarray, NoneType]): return numpy array of the image if
            it's downloaded successfullly. Otherwise return None
    """
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


def get_random_image(
    width: int = 400, height: int = 300, save_path: Optional[str] = None
) -> Optional[np.ndarray]:
    """Download a random image with desired size."""
    return download_image(f"https://picsum.photos/{width}/{height}", save_path)
