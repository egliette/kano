import shutil


def remove_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents removed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

