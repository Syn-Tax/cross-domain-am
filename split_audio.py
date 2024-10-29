import json
import torch
import torchaudio
import re
from tqdm import tqdm

from datastructs import Node, Relation, WordSpan

ALIGNMENTS_PATH = "data/Moral Maze/GreenBelt/alignments.json"
ARGUMENT_MAP_PATH = "data/Moral Maze/GreenBelt/argument_map.json"
AUDIO_PATH = "raw_data/Moral Maze/GreenBelt/audio_16000.mp3"

OUT_PATH = "data/Moral Maze/GreenBelt/audio/"

PADDING = 0.1


def clean_text(txt):
    timestamp_regex = r"\[.{0,10}[0-9]+:[0-9]+:[0-9]+\]"
    punctuation_regex = r"[^a-z]"

    transcript = " ".join(
        [
            ":".join(
                l.split(":")[1:]
                if len(l.split(":")) > 1 and not l.startswith("\t")
                else [l]
            )
            .replace("\n", "")
            .replace("\t", "")
            .lower()
            for l in txt.split("\n")
        ]
    )

    transcript = re.sub(timestamp_regex, "", transcript)
    transcript = re.sub(punctuation_regex, " ", transcript)

    return " ".join(transcript.split())


def get_span(locution, alignments):
    locution = locution.split()
    start_ind = -1
    end_ind = -1

    transcript = [a.word for a in alignments]

    loc_len = len(locution)
    f = 0

    for i in (i for i, e in enumerate(transcript) if e == locution[0]):
        f = i
        if transcript[i : i + loc_len] == locution:
            start_ind = i
            end_ind = i + loc_len - 1
            break

    if start_ind == -1 or end_ind == -1:
        print(transcript.index(locution[0]))
        print("locution not found")
        print(start_ind, end_ind)
        print(f)
        return

    return WordSpan(
        " ".join(locution), alignments[start_ind].start, alignments[end_ind].end
    )


def main():
    with open(ALIGNMENTS_PATH, "r") as f:
        alignments = WordSpan.schema().loads(f.read(), many=True)

    with open(ARGUMENT_MAP_PATH, "r") as f:
        argument_map = Node.schema().loads(f.read(), many=True)

    waveform, sample_rate = torchaudio.load(AUDIO_PATH)

    for node in tqdm(argument_map):
        cleaned_loc = clean_text(node.locution)
        span = get_span(cleaned_loc, alignments)

        if not span:
            print(node.locution)
            print(cleaned_loc)

        node_audio = waveform[
            :,
            int((span.start - PADDING) * sample_rate) : int(
                (span.end + PADDING) * sample_rate
            ),
        ]

        torchaudio.save(f"{OUT_PATH}{node.id}.wav", node_audio, sample_rate)


if __name__ == "__main__":
    main()
