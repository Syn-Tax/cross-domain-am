import json
import torch
import torchaudio
import re
from tqdm import tqdm

from datastructs import Node, Relation, Segment

ALIGNMENTS_PATH = "data/Moral Maze/Hypocrisy/alignments.json"
ARGUMENT_MAP_PATH = "data/Moral Maze/Hypocrisy/argument_map.json"
AUDIO_PATH = "raw_data/Moral Maze/Hypocrisy/audio_16000.wav"

OUT_PATH = "data/Moral Maze/Hypocrisy/audio/"

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

    for i in (ind for ind, e in enumerate(transcript) if locution[0] in e):
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

    return Segment(
        " ".join(locution), alignments[start_ind].start, alignments[end_ind].end
    )


def main(alignments_path, argument_map_path, audio_path):
    with open(alignments_path, "r") as f:
        alignments = Segment.schema().loads(f.read(), many=True)

    with open(argument_map_path, "r") as f:
        argument_map = Node.schema().loads(f.read(), many=True)

    waveform, sample_rate = torchaudio.load(audio_path)

    for node in tqdm(argument_map):
        cleaned_loc = clean_text(node.locution)
        span = get_span(cleaned_loc, alignments)

        if not span:
            print(node.locution)
            print(cleaned_loc)
            print(node.relations)
            continue

        node_audio = waveform[
            :,
            int((span.start - PADDING) * sample_rate) : int(
                (span.end + PADDING) * sample_rate
            ),
        ]

        torchaudio.save(f"{OUT_PATH}{node.id}.wav", node_audio, sample_rate)


if __name__ == "__main__":
    main(ALIGNMENTS_PATH, ARGUMENT_MAP_PATH, AUDIO_PATH)
