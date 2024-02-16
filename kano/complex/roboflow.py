import glob
import os
import random
import shutil

import cv2
from tqdm import tqdm

from kano.file_utils import create_folder, split_file_path


def copy_and_rename_file(file_path, target_folder, new_file_name):
    create_folder(target_folder)
    shutil.copy(file_path, target_folder)
    target_file_path = os.path.join(target_folder, os.path.basename(file_path))
    os.rename(target_file_path, os.path.join(target_folder, new_file_name))


def merge_datasets(folders_paths, merged_folder_path="dataset"):
    for folder_path in folders_paths:
        dataset_name = split_file_path(folder_path)[-1]

        # Rename images and labels files according to dataset name
        txt_files = glob.glob(
            os.path.join(folder_path, "**/*.txt"), recursive=True
        )
        jpg_files = glob.glob(
            os.path.join(folder_path, "**/*.jpg"), recursive=True
        )

        txt_files = [f for f in txt_files if "README" not in f]

        all_files = txt_files + jpg_files

        for file_path in tqdm(all_files, desc=dataset_name):
            subfolders = split_file_path(file_path)
            subset = subfolders[-3]
            file_type = subfolders[-2]

            file_name = os.path.basename(file_path)
            new_file_name = f"{dataset_name}_{file_name}"
            target_folder = os.path.join(merged_folder_path, subset, file_type)

            copy_and_rename_file(file_path, target_folder, new_file_name)

    print(f"Saved at {merged_folder_path}")


def split_dataset(
    dataset_path, train_percent=80, valid_percent=10, target_folder=None
):
    if target_folder is None:
        target_folder = dataset_path

    # Calculate test_ratio
    test_percent = 100 - train_percent - valid_percent

    # Get the list of image files
    image_folder = os.path.join(dataset_path, "images")
    image_files = os.listdir(image_folder)
    random.shuffle(image_files)

    # Calculate the number of files for each split
    total_files = len(image_files)
    train_split = int(total_files * train_percent * 0.01)

    if test_percent == 0:
        valid_split = total_files - train_split
    else:
        valid_split = int(total_files * valid_percent * 0.01)

    # Create Train, Valid, and Test folders
    train_folder = os.path.join(target_folder, "train")
    valid_folder = os.path.join(target_folder, "valid")
    test_folder = os.path.join(target_folder, "test")

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(valid_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Create subfolders for images and labels
    for folder in [train_folder, valid_folder, test_folder]:
        os.makedirs(os.path.join(folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(folder, "labels"), exist_ok=True)

    # Copy files to the corresponding folders
    for i, file_name in enumerate(image_files):
        src_image = os.path.join(image_folder, file_name)
        label_file = os.path.join(
            dataset_path, "labels", file_name[:-4] + ".txt"
        )

        if i < train_split:
            dst_folder = train_folder
        elif i < train_split + valid_split:
            dst_folder = valid_folder
        else:
            dst_folder = test_folder

        dst_image = os.path.join(dst_folder, "images", file_name)
        dst_label = os.path.join(dst_folder, "labels", file_name[:-4] + ".txt")

        shutil.copy(src_image, dst_image)
        shutil.copy(label_file, dst_label)


def get_annotated_image(image_path):
    folder_path, image_name = os.path.split(image_path)
    label_folder_path = os.path.join(os.path.dirname(folder_path), "labels")
    label_name = image_name[:-4] + ".txt"
    label_path = os.path.join(label_folder_path, label_name)

    with open(label_path, "r") as label_file:
        lines = label_file.readlines()
        labels = [line.strip().split() for line in lines]

    image = cv2.imread(image_path)
    for label in labels:
        x, y, w, h = map(float, label[1:])
        x, y, w, h = (
            int(x * image.shape[1]),
            int(y * image.shape[0]),
            int(w * image.shape[1]),
            int(h * image.shape[0]),
        )
        cv2.rectangle(
            image,
            (x - w // 2, y - h // 2),
            (x + w // 2, y + h // 2),
            (0, 255, 0),
            2,
        )

    return image
