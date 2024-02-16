# 🦌 Kano - Tools for Computer Vision Tasks

[![Kano CI](https://github.com/egliette/kano/actions/workflows/python-app.yml/badge.svg)](https://github.com/egliette/kano/actions/workflows/python-app.yml)

**Kano** is a Python package providing utility functions for Computer Vision tasks. Its primary focus is simplifying lengthy functions, allowing developers to concentrate more on the main processes.

## 📥 Installation

[The latest released version](https://pypi.org/project/kano-cv/) is available on PyPI. You can install it by running the following command in your terminal:

```bash
pip install kano-cv
```

## 🚀 Usage

- Computer vision tasks:
    - Object Detection (In progress)
    - Object Segmentation (In progress)
    - [YOLO Datasets Splitting/Merging](#yolo-datasets-splittingmerging)
- [Files/Folders Manipulating](#filesfolders-manipulating)
- [Images Processing](#images-processing)
- [Videos Processing](#videos-processing)

### 🗃️ YOLO Datasets Splitting/Merging

**Test these utilities here:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f8H-QzypOlpzA6sUR0WF3IGxtY6xmF1U?usp=sharing)

If you are using Roboflow to label and struggling to merge many Workspaces or YOLO format projects, **Kano** provides a utility to merge them with just one command:

```python
from kano.complex.roboflow import merge_datasets


# Each dataset folder contains two folders "images" and "labels"
folders = [
    "Dataset_1",
    "Dataset_2",
]

merge_datasets(folders, merged_folder_path="dataset")
```

You can also split a dataset into train/valid/test with your own split ratio with only one command:

```python
from kano.complex.roboflow import split_dataset


# Split a dataset (contains "images" and "labels")
# into train, valid, test folders with a given ratio
split_dataset(
    dataset_path="dataset/train",
    train_percent=80,
    valid_percent=20,
    target_folder="splitted_dataset",
)
```

### 📁 Files/Folders Manipulating

**Test these utilities here:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1878V0IPa36bsTwPTk5NQSszF7UAyJ9Wq?usp=sharing)

**Kano** is designed to run many common functions in just one line:

```python
from kano.file_utils import create_folder, print_foldertree, remove_folder

# Create a folder and its subfolder without errors
create_folder("folder_A/subfolder")

for i in range(2):
    with open(f"folder_A/subfolder/file_{i}.txt", "w") as f:
        pass

print_foldertree("folder_A")
# folder_A (2 files)
# |-- subfolder (2 files)

# Remove a folder with its content without errors
remove_folder("folder_A/subfolder")
```

You can even zip many folders (and files) by providing their paths and the destination path in a function call:

```python
zip_paths(["folder_A", "folder_B"], "zipfile.zip")
```

### 🖼️ Images Processing

**Test these utilities here:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/181jZX3PNylk0Ry133e9ZH5k2vlPV9zxW?usp=sharing)

You can quickly download an image using a URL and show it in IPython notebooks or Python files:

```python
from kano.image_utils import download_image, show_image


image = download_image("https://avatars.githubusercontent.com/u/77763935?v=4", "image.jpg")

# using a numpy array
show_image(image)

# using a file path
show_image("image.jpg")
```

or you can get a random image with a specific size:

```python
from kano.image_utils import get_random_picture


image = get_random_picture(width=400, height=300, save_path="random_image.jpg")

# using a numpy array
show_image(image)

# using a file path
show_image("random_image.jpg")
```

### 🎞️ Videos Processing

**Test these utilities here:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pqHmUHHTnmIfACupcLr_em3PdXNnGOkp?usp=sharing)

**Kano** helps you extract images from a video. For demo purposes, I will download a video from YouTube using [pytube](https://github.com/pytube/pytube). If you find this function helpful, please give a star to [the original repo](https://github.com/pytube/pytube).

```python
from kano.video_utils import download_youtube_video, extract_frames


download_youtube_video("https://www.youtube.com/watch?v=<VIDEOID>", "video.mp4")

# Get 1 image per 2 seconds
extract_frames(
    video_path="video.mp4",
    target_folder="frames",
    seconds_interval=2,
)
```

## 🙋‍♂️ Contributing to Kano

All contributions, bug reports, bug fixes, enhancements, and ideas are welcome. Feel free to create pull requests or issues so that we can improve this library together.

## 🔑 License
Kano is licensed under the [MIT](LICENSE) license.
