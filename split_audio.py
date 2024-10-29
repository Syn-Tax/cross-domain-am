import json
import torch
import torchaudio
import re
import tqdm

from datastructs import Node, Relation, WordSpan

ALIGNMENTS_PATH = "data/Moral Maze/GreenBelt/alignments.json"
ARGUMENT_MAP_PATH = "data/Moral Maze/GreenBelt/argument_map.json"
AUDIO_PATH = "raw_data/Moral Maze/GreenBelt/audio_16000.mp3"

OUT_PATH = "data/Moral Maze/GreenBelt/audio/"

PADDING = 0.5


def clean_text(txt):
    timestamp_regex = r"\[.{0,10}[0-9]+:[0-9]+:[0-9]+\]"
    punctuation_regex = r"[^a-z0-9]"

    transcript = " ".join(
        [
            ":".join(l.split(":")[1:] if len(l.split(":")) > 1 else [l])
            .replace("\n", "")
            .replace("\t", "")
            .lower()
            for l in txt.split("\n")
        ]
    )

    transcript = re.sub(timestamp_regex, "", transcript)
    transcript = re.sub(punctuation_regex, " ", transcript)

    return transcript


def get_span(locution, alignments):
    locution = locution.split()
    found_until = -1
    curr_word = locution[0]

    start_ind = -1
    end_ind = -1

    transcript = [a.word for a in alignments]

    loc_len = len(locution)

    for i in (i for i, e in enumerate(transcript) if e == locution[0]):
        if transcript[i : i + loc_len] == locution:
            start_ind = i
            end_ind = i + loc_len
            break

    print(transcript[start_ind:end_ind])

    if start_ind == -1 or end_ind == -1:
        print("locution not found")
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

    for node in argument_map:
        cleaned_loc = clean_text(node.locution)
        span = get_span(cleaned_loc, alignments)

        node_audio = waveform[
            :,
            int((span.start - PADDING) * sample_rate) : int(
                (span.end + PADDING) * sample_rate
            ),
        ]

        torchaudio.save(f"{OUT_PATH}{node.id}.wav", node_audio, sample_rate)
        return


if __name__ == "__main__":
    main()
