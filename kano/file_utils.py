import os
import re
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path

VERSION_OPERATORS = [
    "==",
    "===",
    "~=",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    # 'and', 'or', 'in', 'not in',
]

IGNORE_START_WITH = ('"', "#", "-", "git+")


def generate_random_filename():
    random_uuid = uuid.uuid4()
    return str(random_uuid)


def get_size(path, unit="bytes"):
    path = Path(path)

    if path.is_file():
        size = path.stat().st_size
    elif path.is_dir():
        total_size = 0
        for sub_path in path.glob("**/*"):
            if sub_path.is_file():
                total_size += sub_path.stat().st_size
        size = total_size
    else:
        raise ValueError("Path is neither a file nor a directory.")

    if unit == "bytes":
        return size
    elif unit == "KB":
        return size / 1024
    elif unit == "MB":
        return size / (1024**2)
    elif unit == "GB":
        return size / (1024**3)
    else:
        raise ValueError("Invalid unit")


def list_files(folder_path, return_absolute_paths=True):
    """
    List files in a folder.

    Args:
        folder_path (str): Path to the folder.
        return_absolute_paths (bool): Whether to return absolute paths or relative paths.

    Returns:
        sorted_files_paths (list): A list of file paths.
    """
    folder_path = Path(folder_path)
    items = list(folder_path.iterdir())

    files = [item.resolve() for item in items if item.is_file()]
    sorted_files = sorted(files)

    if not return_absolute_paths:
        sorted_files = [file.relative_to(folder_path) for file in sorted_files]

    sorted_files_paths = [str(file.as_posix()) for file in sorted_files]

    return sorted_files_paths


def list_folders(folder_path, return_absolute_paths=True):
    """
    List folders in a folder.

    Args:
        folder_path (str): Path to the folder.
        return_absolute_paths (bool): Whether to return absolute paths or relative paths.

    Returns:
        sorted_folders_paths (list): A list of folder paths.
    """
    folder_path = Path(folder_path)
    items = list(folder_path.iterdir())

    folders = [item.resolve() for item in items if item.is_dir()]

    sorted_folders = sorted(folders)

    if not return_absolute_paths:
        sorted_folders = [
            folder.relative_to(folder_path) for folder in sorted_folders
        ]

    sorted_folders_paths = [str(file.as_posix()) for file in sorted_folders]

    return sorted_folders_paths


def list_contents(folder_path, return_absolute_paths=True):
    return list_files(folder_path, return_absolute_paths), list_folders(
        folder_path, return_absolute_paths
    )


def get_folder_details(folder_path):
    files_paths, folders_paths = list_contents(folder_path)
    files_count, folders_count = len(files_paths), len(folders_paths)
    return f"({files_count} files + {folders_count} folders)"


def print_foldertree(folder_path, level=0, max_level=1, verbose=True):
    """
    Print the folder tree structure.

    Args:
        folder_path (str): Path to the folder.
        level (int): Current level to apply recursion, you do not need to provide this field.
        max_level (int): Maximum level of folder tree.
        verbose (bool): Whether to print additional details.
    """
    if level == 0:
        current_line = f"{folder_path} "
        if verbose:
            current_line += get_folder_details(folder_path)
        print(current_line)

    files_paths, child_folders_paths = list_contents(folder_path)

    if len(files_paths) > 0:
        print("|   " * (level + 1))

    for file_path in files_paths:
        file_name = Path(file_path).name
        current_line = "|   " * level + f"|-- {file_name} "
        if verbose:
            file_size = get_size(str(file_path), "KB")
            current_line += f"({file_size:.2f} KB)"
        print(current_line)

    if len(child_folders_paths) > 0:
        print("|   " * (level + 1))

    for i, child_folder_path in enumerate(child_folders_paths):
        child_folder_name = Path(child_folder_path).name
        current_line = "|   " * level + f"|-- {child_folder_name} "
        if verbose:
            current_line += get_folder_details(child_folder_path)
        print(current_line)
        if level + 1 < max_level:
            print_foldertree(child_folder_path, level + 1, verbose)

        if i + 1 < len(child_folders_paths):
            print("|   " * (level + 1))


def zip_paths(paths, output_zip):
    """
    Create a ZIP file containing specified paths.

    Args:
        paths (list): List of file or folder paths to be included in the ZIP file.
        output_zip (str): Path to save the output ZIP file.
    """
    with zipfile.ZipFile(output_zip, "w") as zipf:
        for path in paths:
            if os.path.isdir(path):
                folder_path = path
                for folder_root, _, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(folder_root, file)
                        arcname = os.path.relpath(file_path, folder_path)
                        zipf.write(
                            file_path,
                            arcname=os.path.join(
                                os.path.basename(folder_path), arcname
                            ),
                        )
            elif os.path.isfile(path):
                file_path = path
                filename = os.path.basename(file_path)
                zipf.write(file_path, arcname=filename)


def remove_folder(folder_path):
    """
    Remove a folder and its contents.

    Args:
        folder_path (str): Path to the folder to be removed.
    """
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents removed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def create_folder(folder_path):
    """
    Create a folder if it doesn't exist.

    Args:
        folder_path (str): Path to the folder to be created.
    """
    os.makedirs(folder_path, exist_ok=True)


def split_file_path(file_path):
    folders = []
    path = file_path
    while True:
        path, folder = os.path.split(path)
        if folder:
            folders.insert(0, folder)
        else:
            break

    return folders


def get_package_names(requirements_file_path):
    package_names = []
    with open(requirements_file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith(IGNORE_START_WITH):
                package_name = line
                for op in VERSION_OPERATORS:
                    if op in line:
                        package_name = re.split(
                            f"\s*{re.escape(op)}\s*", package_name, maxsplit=1
                        )[0]

                package_names.append(package_name.strip())

    return package_names


def get_installed_versions(package_names):
    installed_versions = {}
    for package_name in package_names:
        result = subprocess.run(
            ["pip", "show", package_name], capture_output=True, text=True
        )
        output = result.stdout.strip()
        lines = output.split("\n")
        version_line = [line for line in lines if line.startswith("Version:")]

        if version_line:
            version = version_line[0].split(": ")[1].strip()
            installed_versions[package_name] = version
        else:
            installed_versions[package_name] = None

    return installed_versions


def print_package_versions(requirements_file_path="requirements.txt"):
    """
    Print packages listed in requirements file and their version in the current environment

    Args:
        requirements_file_path (str): path to a file contain packages names
    """
    package_names = get_package_names(requirements_file_path)
    installed_versions = get_installed_versions(package_names)

    for package_name, version in installed_versions.items():
        if version is None:
            print(f"# {package_name}")
        else:
            print(f"{package_name}=={version}")
