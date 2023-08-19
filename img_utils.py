import requests
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

 
def download_image(url, save_path=None):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            img_data = BytesIO(response.content)
            img = Image.open(img_data)
            img_array = np.array(img)
            
            if save_path:
                img.save(save_path)
            
            return img_array
        else:
            print(f"Failed to download image from {url}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def load_image_as_numpy(image_path, target_size=None):
    try:
        img = Image.open(image_path)
        if target_size:
            img = img.resize(target_size)
        img_array = np.array(img)
        return img_array
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def plot_image(image, title=None):
    try:
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        plt.figure()
        plt.imshow(image)
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")
