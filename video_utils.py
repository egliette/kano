from pytube import YouTube
from IPython.display import HTML
from base64 import b64encode
import cv2
import numpy as np


def download_youtube_video(url, filename):
    yt = YouTube(url)
    video_stream = yt.streams.get_highest_resolution()
    video_stream.download(filename=filename)


def play_movie_ipynb(path):
    mp4 = open(path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    HTML("""
    <video width=400 controls>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url)

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
