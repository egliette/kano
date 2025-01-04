import copy
import os

import cv2
import numpy as np
import tqdm

from kano.file_utils import create_folder
from kano.image import concatenate_images


def get_frame_at_second(video_path, target_second):
    """
    Get numpy array of a frame from a video at the given second

    Args:
        video_path (str): path of the video
        target_second (int): second timestamp to get frame

    Returns:
        frame (np.ndarray): numpy array of the frame at the given second
    """
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
    """
    Extract frames from a video with a given seconds interval

    Args:
        video_path (str): path of the video
        target_folder (str): path to save extracted frames
        seconds_interval (float): amount of seconds between two extracted frames
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps * seconds_interval)

    create_folder(target_folder)

    max_length = len(str(frame_count))
    for frame_number in tqdm.tqdm(range(0, frame_count, frame_interval)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break
        number = str(frame_number).zfill(max_length)
        frame_filename = os.path.join(target_folder, f"frame_{number}.jpg")
        cv2.imwrite(frame_filename, frame)

    cap.release()


def cut_video(input_video_path, output_video_path, start_second, end_second):
    """
    Cut a segment from a video file and save it as a new video.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video file.
        start_second (float): Start time of the segment to be cut in seconds.
        end_second (float): End time of the segment to be cut in seconds.
    """
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_second * fps)
    end_frame = int(end_second * fps)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()


def add_title(
    image, title, font_scale=2, font_thickness=3, title_padding_size=10
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = image.shape[0] + title_padding_size + text_size[1]
    height_diff = title_padding_size + text_size[1] + title_padding_size
    padding = np.zeros((height_diff, image.shape[1], 3), dtype=np.uint8)
    padded_image = np.vstack([image, padding])
    cv2.putText(
        padded_image,
        title,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
    )
    return padded_image


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration


def concatenate_videos(
    video_paths,
    titles=None,
    output_video_path="concatenated_video.mp4",
    total_seconds=None,
    font_scale=2,
    font_thickness=3,
    title_padding_size=10,
    frame_padding_size=10,
):
    """
    Concatenate multiple video files into a single video.

    Args:
        video_paths (list): A 2D list of video file paths to be concatenated.
        titles (list): A 2D list of titles corresponding to each video segment.
        output_video_path (str): Path to save the concatenated video file.
        total_seconds (float): Total duration of the output video in seconds.
        font_scale (int): Font scale for titles.
        font_thickness (int): Font thickness for titles.
        title_padding_size (int): Padding size for titles.
        frame_padding_size (int): Padding size between frames.
    """
    # ensure video_paths is a 2D list
    new_video_paths = copy.deepcopy(video_paths)
    if not isinstance(video_paths[0], list):
        new_video_paths = [copy.deepcopy(video_paths)]

    rows = len(new_video_paths)
    cols = max(len(row) for row in new_video_paths)

    for video_path in new_video_paths:
        video_path += [None] * (cols - len(video_path))

    # get target video duration
    video_captures = list()
    durations = list()

    max_width = 0
    max_height = 0
    sample_cap = None
    for i, row in enumerate(new_video_paths):
        video_captures.append(list())
        for video_path in row:
            if video_path is not None:
                sample_cap = cv2.VideoCapture(video_path)
                width = int(sample_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(sample_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                max_width = max(width, max_width)
                max_height = max(height, max_height)
                video_captures[i].append(sample_cap)
                durations.append(get_video_duration(video_path))
            else:
                video_captures[i].append(None)

    min_duration = min(durations)
    if total_seconds is not None:
        min_duration = min(min_duration, total_seconds)

    # init output video
    fps = int(sample_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize("SAMPLE", font, font_scale, font_thickness)[0]

    output_width = max_width * cols + frame_padding_size * (cols - 1)
    output_height = (
        max_height + (text_size[1] + title_padding_size * 2)
    ) * rows + frame_padding_size * (rows - 1)

    output_video = cv2.VideoWriter(
        output_video_path, fourcc, fps, (output_width, output_height)
    )
    num_frames = int(min_duration * fps)

    for __ in tqdm.tqdm(range(num_frames), desc="Videos concatenating"):
        frames = list()
        for row in range(rows):
            frames.append(list())
            for col in range(cols):
                cap = video_captures[row][col]
                if (
                    len(titles) < row + 1
                    or len(titles[row]) < col + 1
                    or titles[row][col] is None
                ):
                    title = " "
                else:
                    title = titles[row][col]

                frame = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                if cap is not None:
                    __, frame = cap.read()

                frame = add_title(
                    frame,
                    title,
                    font_scale,
                    font_thickness,
                    title_padding_size,
                )

                frames[row].append(frame)

        concatenated_frame = concatenate_images(frames, frame_padding_size)

        output_video.write(concatenated_frame)

    for row_caps in video_captures:
        for cap in row_caps:
            if cap is not None:
                cap.release()

    output_video.release()
