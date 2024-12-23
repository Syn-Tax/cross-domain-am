import json
import torch
import torchaudio
import re
from tqdm import tqdm
import pickle

from datastructs import Node, Relation, Segment
import alignment

QT_EPISODE = "03.18June2020"

ALIGNMENTS_PATH = f"data/Question Time/{QT_EPISODE}/alignments.json"
ARGUMENT_MAP_PATH = f"data/Question Time/{QT_EPISODE}/argument_map.json"
AUDIO_PATH = f"raw_data/Question Time/{QT_EPISODE}/audio.wav"

OUT_PATH = f"data/Question Time/{QT_EPISODE}/audio/"

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


def get_span(locution, alignments, node):
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
        if len(node.relations) == 0:
            return
        try:
            print(transcript.index(locution[0]))
        except:
            pass
        print("locution not found")
        print(start_ind, end_ind)
        print(f)
        return

    return Segment(
        " ".join(locution), alignments[start_ind].start, alignments[end_ind].end
    )


def main(alignments_path, argument_map_path, audio_path):
    with open(argument_map_path, "r") as f:
        argument_map = Node.schema().loads(f.read(), many=True)

    waveform, sample_rate = torchaudio.load(audio_path)

    bundle = torchaudio.pipelines.MMS_FA

    print("############ Calculating Emissions ################")
    emissions = alignment.calculate_emissions(waveform, sample_rate, bundle)

    # with open("emissions.pkl", "wb+") as f:
    #     pickle.dump(emissions, f)
    # with open("emissions.pkl", "rb+") as f:
    #     emissions = pickle.load(f)

    span_scores = []
    print("############ Getting Alignments ################")

    for i, node in enumerate(tqdm(argument_map)):
        cleaned_loc = clean_text(node.locution)

        span = alignment.get_span(emissions, bundle, cleaned_loc, waveform.size(1), sample_rate)
        span_scores.append(span.score)

        argument_map[i].audio_score = span.score

        node_audio = waveform[
            :,
            int((span.start - PADDING) * sample_rate) : int(
                (span.end + PADDING) * sample_rate
            ),
        ]

        torchaudio.save(f"{OUT_PATH}{node.id}.wav", node_audio, sample_rate)

    with open(argument_map_path, "w") as f:
        out = Node.schema().dumps(argument_map, many=True)
        f.write(out)
        print("written argument map with audio scores")

    print(f"Mean audio confidence score: {sum(span_scores) / len(span_scores)}")


if __name__ == "__main__":
    main(ALIGNMENTS_PATH, ARGUMENT_MAP_PATH, AUDIO_PATH)
