# import json
import tqdm
import ffmpeg
import os

VIDEO_PATH = "raw_data/Question Time/episodes"
AUDIO_PATH = "raw_data/Question Time"

files = [f for f in os.listdir(VIDEO_PATH) if os.path.isfile(os.path.join(VIDEO_PATH, f))]

for file in tqdm.tqdm(files):
    out_path = ".".join(file.split(".")[:2])
    (
        ffmpeg.input(os.path.join(VIDEO_PATH, file))
        .output(os.path.join(AUDIO_PATH, out_path, "audio.wav"), ar="16k", ac=1, log_level="warning")
        .run(overwrite_output=True)
    )
