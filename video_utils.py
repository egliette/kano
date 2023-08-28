from pytube import YouTube
from IPython.display import HTML
from base64 import b64encode


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

