import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from kano.detect_utils import draw_bbox, xywh2xyxy
from kano.file_utils import create_folder, list_files
from kano.image_utils import concatenate_images, show_image
from kano.pose_utils import draw_skeleton

TASKS = ["detect", "pose"]


class YoloImage:
    """
    Represents an image annotated in YOLO format, which includes bounding boxes or skeletons.

    Attributes:
        image (numpy.ndarray): The original image.
        image_path (str): Path to the image file.
        label_path (str): Path to the label file corresponding to the image.
        task (str): Task type, either "detect" or "pose".
        labels (list): List of dictionaries containing label information.
        labels_dict (dict): Dictionary mapping class IDs to class names.
    """

    def __init__(self, image_path, labels_dict=None, task="detect"):
        """
        Initialize a YoloImage object.

        Args:
            image_path (str): Path to the image file.
            labels_dict (dict): Dictionary mapping class IDs to class names.
            task (str): Task type. Possible values: "detect", "pose".
        """
        self.image = cv2.imread(image_path)
        self.image_path = image_path
        self.label_path = self.get_label_path(image_path)
        if task not in TASKS:
            raise ValueError("Unexpected task. Please provide one of:", TASKS)
        self.task = task
        self.labels = self.get_labels(self.label_path)
        self.labels_dict = labels_dict

    def get_label_path(self, image_path):
        """
        Get the path to the label file corresponding to the given image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            label_path (str): Path to the label file.
        """
        image_path = Path(image_path)
        images_folder_path = image_path.parent
        dataset_path = images_folder_path.parent
        labels_folder_path = dataset_path / "labels"
        label_filename = image_path.with_suffix(".txt").name
        label_path = labels_folder_path / label_filename

        return str(label_path)

    def get_labels(self, label_path):
        """
        Parse the label file and extract label information.
        Each label can contain:
            - class: class of the object
            - s_xywh: scaled xywh box
            - xyxy: xyxy box
            - keypoints: list of dict(xy, state) for pose estimation tasks

        Args:
            label_path (str): Path to the label file.

        Returns:
            labels (list): List of dictionaries, each containing label information.
        """
        labels = list()
        image_height, image_width = self.image.shape[:2]
        with open(label_path, "r") as file:
            for line in file:
                line = line.strip().split()
                label = {
                    "class": int(line[0]),
                    "s_xywh": np.array([float(x) for x in line[1:5]]),
                }
                xywh = label["s_xywh"].copy() * np.array(
                    [image_width, image_height, image_width, image_height]
                )
                label["xyxy"] = xywh2xyxy(xywh)

                if self.task == "pose":
                    keypoints = list()
                    for start_id in range(5, len(line), 3):
                        # "state" in [0, 1, 2] with:
                        # - 0: deleted
                        # - 1: occluded
                        # - 2: visible
                        x = int(float(line[start_id]) * image_width)
                        y = int(float(line[start_id + 1]) * image_height)
                        state = int(float(line[start_id + 2]))
                        keypoints.append(
                            {
                                "xy": (x, y),
                                "state": state,
                            }
                        )
                    label["keypoints"] = keypoints

                labels.append(label)
        return labels

    def show_image(self, figsize=(10, 10)):
        """
        Display the original image.

        Args:
            figsize (tuple(int, int)): Size of the figure (width, height) in inches.
        """
        show_image(self.image, figsize)

    def get_annotated_image(self):
        """
        Get the annotated image with bounding boxes or skeletons drawn.

        Returns:
            annotated_image (np.ndarray): Annotated image.
        """
        annotated_image = self.image.copy()
        for label in self.labels:

            if self.task == "pose":
                annotated_image = draw_skeleton(
                    annotated_image, label["keypoints"]
                )

            cls = label["class"]
            if self.labels_dict is not None:
                cls = self.labels_dict[cls]
            bbox = label["s_xywh"]

            annotated_image = draw_bbox(
                annotated_image, bbox, "s_xywh", (0, 255, 0), str(cls)
            )

        return annotated_image

    def show_annotated_image(self, figsize=(10, 10)):
        """
        Display the annotated image with bounding boxes or skeletons drawn.

        Returns:
            figsize (tuple(int, int)): Size of the figure (width, height) in inches.
        """
        annotated_image = self.get_annotated_image()
        show_image(annotated_image, figsize)

    def copy_to(
        self,
        target_folder_path,
        prefix="",
        reindex_dict=None,
        target_stem=None,
    ):
        """
        Copy the image and its label file to the specified target folder.

        Returns:
            target_folder_path (str): Path to the target folder.
            prefix (str): Prefix to be added to the filenames
            reindex_dict (dict): Dictionary mapping original class IDs to new class IDs.
            target_stem (str): Name stem for the copied files
        """
        source_image_path = Path(self.image_path)
        source_label_path = Path(self.label_path)
        target_folder_path = Path(target_folder_path)

        if target_stem is None:
            target_image_path = (
                target_folder_path
                / "images"
                / (prefix + source_image_path.name)
            )
            target_label_path = (
                target_folder_path
                / "labels"
                / (prefix + source_label_path.name)
            )
        else:
            target_image_path = (
                target_folder_path / "images" / (target_stem + ".jpg")
            )
            target_label_path = (
                target_folder_path / "labels" / (target_stem + ".txt")
            )

        shutil.copyfile(self.image_path, str(target_image_path))
        shutil.copyfile(self.label_path, str(target_label_path))

        if reindex_dict:
            with open(str(source_label_path), "r") as f:
                lines = f.readlines()

            new_lines = list()
            for line in lines:
                class_id = int(line.split()[0])
                if class_id in reindex_dict:
                    new_class_id = reindex_dict[class_id]
                    if new_class_id is not None:
                        new_line = (
                            f"{new_class_id} {' '.join(line.split()[1:])}\n"
                        )
                        new_lines.append(new_line)

            with open(str(target_label_path), "w") as f:
                f.writelines(new_lines)


class YoloDataset:
    """
    Dataset class with Yolo format.

    Attributes:
        dataset_path (Path): path to the dataset folder
        name (str): name of the dataset (folder name)
        classes (list(str)): names of classes in the dataset
        task (str): dataset usecase, must be a task in ["detect", "pose"]
    """

    def __init__(self, dataset_path, task="detect"):
        """
        Initialize a YoloDataset object.

        Returns:
            dataset_path (str): Path to the dataset folder.
            task (str): . Dataset use case. Default is "detect".
                        Possible values: "detect", "pose".
        """
        self.dataset_path = Path(dataset_path)
        self.name = self.dataset_path.name
        self.train_folder = self.dataset_path / "train"
        self.valid_folder = self.dataset_path / "valid"
        self.test_folder = self.dataset_path / "test"
        self.classes = self.get_classes(str(self.dataset_path / "data.yaml"))
        if task not in TASKS:
            raise ValueError("Unexpected task. Please provide one of:", TASKS)
        self.task = task

    @classmethod
    def get_classes(cls, yaml_path):
        """
        Retrieve classes from a YAML file.

        Returns:
            yaml_path (str): Path to the YAML file containing class names.

        Returns:
            names (list(str)): List of class names.
        """
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)
        return data["names"]

    def summary(self):
        """
        Print summary information about the dataset.
        """
        print(f"Summary dataset {self.dataset_path.name}:")
        print("- Classes: ", self.classes)
        print("- Subsets:")
        total_file_count = 0
        for folder_path in [
            self.train_folder,
            self.valid_folder,
            self.test_folder,
        ]:
            if folder_path.exists():
                file_count = len(list_files(str(folder_path / "images")))
                print(f"  + {folder_path.name}: {file_count} images")
                total_file_count += file_count
        print("- Total images:", total_file_count)

    @classmethod
    def _combine_classes(cls, datasets_paths):
        """
        Combine classes from multiple datasets.

        Returns:
            datasets_paths (list(str)): Paths to the dataset folders.

        Returns:
            classes (list(str)): Combined list of class names.
        """
        classes = set()

        for path in datasets_paths:
            new_classes = cls.get_classes(str(Path(path) / "data.yaml"))
            classes.update(new_classes)

        classes = list(classes)
        classes.sort()

        return classes

    @classmethod
    def _get_reindex_dict(cls, source_classes, target_classes):
        """
        Get the reindexing dictionary for mapping classes from source to target.

        Returns:
            source_classes (list(str)): Source class names.
            target_classes (list(str)): Target class names.

        Returns:
            reindex_dict (dict): Reindexing dictionary.
        """
        reindex_dict = dict()
        for i, class_name in enumerate(source_classes):
            if class_name in target_classes:
                reindex_dict[i] = target_classes.index(class_name)

        return reindex_dict

    @classmethod
    def _create_simple_yaml_file(cls, dataset_path, classes):
        """
        Create a simple YAML file containing dataset information.

        Returns:
            dataset_path (str): Path to the dataset folder.
            classes (list[str]): List of class names.
        """
        data = {
            "train": "train",
            "val": "valid",
            "test": "test",
            "names": classes,
        }

        yaml_path = Path(dataset_path) / "data.yaml"
        with open(str(yaml_path), "w") as f:
            yaml.dump(data, f)

    @classmethod
    def merge_datasets(cls, datasets_paths, merged_dataset_path):
        """
        Merge multiple datasets into one.

        Returns:
            datasets_paths (list[str]): Paths to the dataset folders to be merged.
            merged_dataset_path (str): Path to the merged dataset folder.
        """
        merged_dataset_path = Path(merged_dataset_path)
        merged_classes = cls._combine_classes(datasets_paths)
        print("Input datasets:")
        for path in datasets_paths:
            dataset = cls(path)
            dataset.summary()
            classes = dataset.classes
            reindex_dict = cls._get_reindex_dict(classes, merged_classes)

            for subset_path in [
                dataset.train_folder,
                dataset.valid_folder,
                dataset.test_folder,
            ]:
                if subset_path.exists():
                    target_subset_path = merged_dataset_path / subset_path.name
                    create_folder(target_subset_path / "images")
                    create_folder(target_subset_path / "labels")
                    images_paths = list_files(str(subset_path / "images"))
                    for image_path in images_paths:
                        yolo_image = YoloImage(image_path)
                        yolo_image.copy_to(
                            target_subset_path,
                            dataset.name + "_",
                            reindex_dict,
                        )

        cls._create_simple_yaml_file(str(merged_dataset_path), merged_classes)

        print("Merged dataset:")
        dataset = cls(str(merged_dataset_path))
        dataset.summary()

    def split(self, splitted_dataset_path, ratios=[0.9]):
        """
        Split the dataset into train, validation, and test subsets.

        Returns:
            splitted_dataset_path (str): Path to the folder where the splitted dataset will be saved.
            ratios (list[float]): Ratios for train, validation, and test subsets. Default is [0.9].
        """
        self.summary()

        images_paths = list()
        for folder_path in [
            self.train_folder,
            self.valid_folder,
            self.test_folder,
        ]:
            if folder_path.exists():
                images_paths += list_files(str(folder_path / "images"))

        random.shuffle(images_paths)

        total_paths = len(images_paths)

        train_count = int(total_paths * ratios[0])
        train_paths = images_paths[:train_count]

        if len(ratios) == 1:
            valid_paths = images_paths[train_count:]
            test_paths = list()
        else:
            valid_count = int(total_paths * ratios[1])
            valid_paths = images_paths[train_count : train_count + valid_count]
            test_paths = images_paths[train_count + valid_count :]

        splitted_dataset_path = Path(splitted_dataset_path)

        subsets = [
            ("train", train_paths),
            ("valid", valid_paths),
            ("test", test_paths),
        ]
        for subset_name, paths in subsets:
            for path in paths:
                yolo_image = YoloImage(path)
                old_subset_name = Path(path).parent.parent.name
                target_folder_path = splitted_dataset_path / subset_name
                create_folder(target_folder_path / "images")
                create_folder(target_folder_path / "labels")
                yolo_image.copy_to(
                    target_folder_path, f"{self.name}_{old_subset_name}_"
                )

        self._create_simple_yaml_file(str(splitted_dataset_path), self.classes)
        YoloDataset(str(splitted_dataset_path)).summary()

    def rename_classes(self, renamed_dataset_path, renaming_dict):
        """
        Rename classes in the dataset.

        Returns:
            renamed_dataset_path (str): Path to the folder where the renamed dataset will be saved.
            renaming_dict (dict): Dictionary mapping original class names to new class names.
        """

        self.summary()

        target_classes = [
            class_name
            for class_name in renaming_dict.values()
            if class_name is not None
        ]

        for class_name in self.classes:
            if class_name not in renaming_dict.keys():
                target_classes.append(class_name)

        target_classes = list(set(target_classes))
        target_classes.sort()

        print("Classes after renaming:", target_classes)

        reindex_dict = dict()

        for i, class_name in enumerate(self.classes):
            renamed_class = class_name
            if class_name in renaming_dict:
                renamed_class = renaming_dict[class_name]

            if renamed_class is None:
                reindex_dict[i] = None
            else:
                reindex_dict[i] = target_classes.index(renamed_class)

        renamed_dataset_path = Path(renamed_dataset_path)
        for subset_name, folder_path in [
            ("train", self.train_folder),
            ("valid", self.valid_folder),
            ("test", self.test_folder),
        ]:
            if folder_path.exists():
                images_paths = list_files(folder_path / "images")
                for path in images_paths:
                    yolo_image = YoloImage(path)
                    target_folder_path = renamed_dataset_path / subset_name
                    create_folder(target_folder_path / "images")
                    create_folder(target_folder_path / "labels")
                    yolo_image.copy_to(
                        target_folder_path, f"{self.name}_", reindex_dict
                    )

        self._create_simple_yaml_file(
            str(renamed_dataset_path), target_classes
        )
        YoloDataset(str(renamed_dataset_path)).summary()

    def show_sample(self, figsize=(10, 10)):
        """
        Show a sample of annotated images from the dataset.

        Returns:
            figsize (tuple(int, int)): Size of the figure (width, height) in inches.
        """

        images_paths = list()
        for folder_path in [
            self.train_folder,
            self.valid_folder,
            self.test_folder,
        ]:
            if folder_path.exists():
                images_paths += list_files(str(folder_path / "images"))

        random.shuffle(images_paths)

        labels_dict = {
            i: class_name for i, class_name in enumerate(self.classes)
        }
        annotated_images = list()
        if len(images_paths) > 9:
            for i in range(3):
                annotated_images.append(list())
                for j in range(3):
                    yolo_image = YoloImage(
                        images_paths[i * 3 + j], labels_dict, self.task
                    )
                    annotated_images[i].append(
                        yolo_image.get_annotated_image()
                    )
        else:
            total_images = min(3, len(images_paths))
            for i in range(total_images):
                yolo_image = YoloImage(images_paths[i], labels_dict, self.task)
                annotated_images.append(yolo_image.get_annotated_image())

        concatenated_images = concatenate_images(annotated_images)

        show_image(concatenated_images, figsize=figsize)

    def number_filenames(self, numbered_dataset_path):
        """
        Number filenames of images in the dataset.

        Returns:
            numbered_dataset_path (str): Path to the folder where the numbered dataset will be saved.
        """
        self.summary()

        total_images = 0

        for folder_path in [
            self.train_folder,
            self.valid_folder,
            self.test_folder,
        ]:
            if folder_path.exists():
                total_images += len(list_files(str(folder_path / "images")))

        n_digits = len(str(total_images))
        i = 0

        subsets = [
            ("train", self.train_folder),
            ("valid", self.valid_folder),
            ("test", self.test_folder),
        ]
        numbered_dataset_path = Path(numbered_dataset_path)
        for subset_name, folder_path in subsets:
            if folder_path.exists():
                images_paths = list_files(str(folder_path / "images"))
                for path in images_paths:
                    yolo_image = YoloImage(path)
                    target_folder_path = numbered_dataset_path / subset_name
                    create_folder(target_folder_path / "images")
                    create_folder(target_folder_path / "labels")
                    yolo_image.copy_to(
                        target_folder_path, target_stem=str(i).zfill(n_digits)
                    )
                    i += 1

        self._create_simple_yaml_file(str(numbered_dataset_path), self.classes)
        YoloDataset(str(numbered_dataset_path)).summary()
