import os

import cv2
from pytube import YouTube
from tqdm import tqdm

from kano.file_utils import create_folder


def download_youtube_video(url, filename):
    yt = YouTube(url)
    video_stream = yt.streams.get_highest_resolution()
    video_stream.download(filename=filename)


def get_frame_at_second(video_path, target_second):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Video file could not be opened.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(target_second * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ret, frame = cap.read()

    cap.release()

    if ret:
        return frame
    else:
        raise ValueError("Frame not found at the specified second.")


def extract_frames(video_path, target_folder, seconds_interval):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps * seconds_interval)

    create_folder(target_folder)

    max_length = len(str(frame_count))
    for frame_number in tqdm(range(0, frame_count, frame_interval)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break
        number = str(frame_number).zfill(max_length)
        frame_filename = os.path.join(target_folder, f"frame_{number}.jpg")
        cv2.imwrite(frame_filename, frame)

    cap.release()
