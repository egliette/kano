import os
import shutil
import zipfile


def count_files_in_folder(folder_path):
    total_files = sum([len(files) for _, _, files in os.walk(folder_path)])
    return total_files


def print_foldertree(root_path, level=0):
    if level == 0:
        print(f"{root_path} ({count_files_in_folder(root_path)} files)")

    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)

        if os.path.isdir(item_path):
            print(
                "|   " * level
                + f"|-- {item} ({count_files_in_folder(item_path)} files)"
            )
            print_foldertree(item_path, level + 1)


def zip_paths(paths, output_zip):
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
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents removed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def create_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)


def list_files_in_folder(folder_path, keep_folder_path=True):
    files_list = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            if keep_folder_path:
                files_list.append(file_path)
            else:
                files_list.append(file_name)

    files_list.sort()

    return files_list


def list_folders_in_folder(folder_path, keep_folder_path=True):
    folders_list = []

    for folder_name in os.listdir(folder_path):
        folder_path_entry = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_path_entry):
            if keep_folder_path:
                folders_list.append(folder_path_entry)
            else:
                folders_list.append(folder_name)

    folders_list.sort()

    return folders_list


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


def get_size(path, unit="bytes"):
    if os.path.isfile(path):
        size = os.path.getsize(path)
    elif os.path.isdir(path):
        size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(path)
            for filename in filenames
        )
    else:
        raise ValueError("Invalid path")

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
