from yt_dlp import YoutubeDL
import multiprocessing
import json

""" File to download QT episodes using yt-dlp
"""

with open("../QTepisodes.json", "r") as f:
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
