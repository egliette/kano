import os
import shutil


def remove_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents removed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

def create_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)