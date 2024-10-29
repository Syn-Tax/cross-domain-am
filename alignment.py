import json
import torch
import torchaudio
import re

AUDIO_PATH = "raw_data/Moral Maze/GreenBelt/audio_8000.wav"
TRANSCRIPT_PATH = "raw_data/Moral Maze/GreenBelt/transcript.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def align():
    waveform, _ = torchaudio.load(AUDIO_PATH)

    with open(TRANSCRIPT_PATH, "r") as f:
        lines = f.readlines()

    timestamp_regex = r"\[.{0,10}[0-9]+:[0-9]+:[0-9]+\]"

    transcript = " ".join(
        [
            ":".join(l.split(":")[1:] if len(l.split(":")) > 1 else [l])
            .replace("\n", "")
            .replace("\t", "")
            for l in lines
        ]
    )

    transcript = re.sub(timestamp_regex, "", transcript).split()

    bundle = torchaudio.pipelines.MMS_FA

    model = bundle.get_model(with_star=False).to(device)
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))

    labels = bundle.get_labels(star=None)
    dictionary = bundle.get_dict(star=None)

    for k, v in dictionary.get_items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    align()
