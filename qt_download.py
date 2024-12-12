from yt_dlp import YoutubeDL
from tqdm import tqdm
import multiprocessing
import json

with open("QTepisodes.json", "r") as f:
    data = json.load(f)

def download(args):
    name, url = args
    if url == "": return
    ydl_opts = {
        "merge-output-format": "mp4",
        "outtmpl": f"episodes/{name}.mp4",
        "writesubtitles": True,
        "subtitlesformat": "srt",
        "subtitleslangs": "en"
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

pool = multiprocessing.Pool(4)
pool.map(download, data.items())
