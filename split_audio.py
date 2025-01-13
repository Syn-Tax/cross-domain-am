import json
import torch
import torchaudio
import re
from tqdm import tqdm
import pickle

from datastructs import Node, Relation, Segment
import audio.alignment as alignment

"""
    Script to split the audio file corresponding to an episode into
    files containing only a single locution
"""

# QT_EPISODE = "30.11November2021"

# ALIGNMENTS_PATH = f"data/Question Time/{QT_EPISODE}/alignments.json"
# ARGUMENT_MAP_PATH = f"data/Question Time/{QT_EPISODE}/argument_map.json"
# AUDIO_PATH = f"raw_data/Question Time/{QT_EPISODE}/audio.wav"

ARGUMENT_MAP_PATH = f"data/Moral Maze/Families/argument_map.json"
AUDIO_PATH = f"raw_data/Moral Maze/Families/audio.wav"

OUT_PATH = f"data/Moral Maze/Families/audio/"

PADDING = 0.1 # number of seconds to include around the locution


def clean_text(txt):
    """Method to clean a transcript

    Args:
        txt (str): text to be cleaned

    Returns:
        str: cleaned text
    """

    # define regular expressions
    timestamp_regex = r"\[.{0,10}[0-9]+:[0-9]+:[0-9]+\]"
    punctuation_regex = r"[^a-z]"

    # process transcript line
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

    # substitute regular expressions
    transcript = re.sub(timestamp_regex, "", transcript)
    transcript = re.sub(punctuation_regex, " ", transcript)

    # return cleaned transcript
    return " ".join(transcript.split())


def main(argument_map_path, audio_path):
    # load argument map
    with open(argument_map_path, "r") as f:
        argument_map = Node.schema().loads(f.read(), many=True)

    # load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    print(sample_rate)

    # load forced alignment model
    bundle = torchaudio.pipelines.MMS_FA

    print("############ Calculating Emissions ################")
    emissions = alignment.calculate_emissions(waveform, sample_rate, bundle)

    span_scores = []
    print("############ Getting Alignments ################")

    # for each node get the audio containing its locution
    for i, node in enumerate(tqdm(argument_map)):
        cleaned_loc = clean_text(node.locution)

        span = alignment.get_span(emissions, bundle, cleaned_loc, waveform.size(1), sample_rate)

        # save confidence scores
        span_scores.append(span.score)
        argument_map[i].audio_score = span.score

        # split audio data    
        node_audio = waveform[
            :,
            int((span.start - PADDING) * sample_rate) : int(
                (span.end + PADDING) * sample_rate
            ),
        ]

        # save to new file containing node id
        torchaudio.save(f"{OUT_PATH}{node.id}.wav", node_audio, sample_rate)

    # write argument map containing alignment confidence scores
    with open(argument_map_path, "w") as f:
        out = Node.schema().dumps(argument_map, many=True)
        f.write(out)
        print("written argument map with audio scores")

    print(f"Mean audio confidence score: {sum(span_scores) / len(span_scores)}")


if __name__ == "__main__":
    main(ARGUMENT_MAP_PATH, AUDIO_PATH)
