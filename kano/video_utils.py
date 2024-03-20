import os
import uuid

import cv2
from moviepy.editor import VideoFileClip
from pytube import YouTube
from tqdm import tqdm

from kano.file_utils import create_folder


def generate_random_filename():
    random_uuid = uuid.uuid4()
    return str(random_uuid)


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


def cut_video(input_video_path, output_video_path, start_second, end_second):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_second * fps)
    end_frame = int(end_second * fps)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    temp_file_path = generate_random_filename() + ".avi"
    out = cv2.VideoWriter(temp_file_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

    clip = VideoFileClip(temp_file_path)
    clip.write_videofile(output_video_path, codec="libx264", fps=clip.fps)

    os.remove(temp_file_path)
