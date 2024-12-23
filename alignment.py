import json
import torch
import torchaudio
import torchaudio.functional as F
import re
from tqdm import tqdm
import math

from datastructs import Segment

QT_EPISODE = "01.28May2020"

AUDIO_PATH = f"raw_data/Question Time/{QT_EPISODE}/audio.wav"
TRANSCRIPT_PATH = f"data/Question Time/{QT_EPISODE}/transcript.txt"

OUT_PATH = f"data/Question Time/{QT_EPISODE}/alignments.json"

CHUNK_LEN = 5


USE_INTEL = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("xpu")


def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]
    scores = scores.exp()
    return alignments, scores


def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret


def save_spans(word_spans, labels, num_frames, waveform_len, sample_rate):
    ratio = waveform_len / num_frames
    output = []
    for word in word_spans:
        word_start = int(word[0].start * ratio) / sample_rate
        word_end = int(word[-1].end * ratio) / sample_rate

        output.append(
            {
                "word": "".join(labels[span.token] for span in word),
                "start": word_start,
                "end": word_end,
            }
        )

    with open(OUT_PATH, "w") as f:
        json.dump(output, f)


def calculate_emissions(waveform, sample_rate, bundle):
    n_splits = math.ceil(waveform.size(1) / (CHUNK_LEN * sample_rate))
    waveform_split = torch.tensor_split(waveform, n_splits, dim=1)

    model = bundle.get_model(with_star=False).to(device)

    emissions = torch.tensor([]).to(device)

    for chunk in tqdm(waveform_split):
        with torch.inference_mode():
            emission, _ = model(chunk.to(device))
            emissions = torch.cat((emissions, emission), dim=1)

    return emissions

def get_span(emissions, bundle, transcript, waveform_len, sample_rate):
    dictionary = bundle.get_dict(star=None)

    dictionary["*"] = len(dictionary)

    star_dim = torch.zeros((1, emissions.size(1), 1), device=emissions.device, dtype=emissions.dtype)
    emissions = torch.cat((emissions, star_dim), 2)

    transcript = transcript.split()
    transcript.insert(0, "*")
    transcript.append("*")

    tokenized_transcript = [dictionary[c] for word in transcript for c in word]

    aligned_tokens, alignment_scores = align(emissions, tokenized_transcript)
    token_spans = F.merge_tokens(aligned_tokens, alignment_scores)
    word_spans = unflatten(token_spans, [len(word) for word in transcript])[1:-1]

    ratio = waveform_len / emissions.size(1)

    transcript_start = int(word_spans[0][0].start * ratio) / sample_rate
    transcript_end = int(word_spans[-1][-1].end * ratio) / sample_rate

    score = sum([s.score for s in token_spans]) / len(token_spans)

    return Segment(
        transcript,
        transcript_start,
        transcript_end,
        score
    )

def main():
    pass
    # with open(TRANSCRIPT_PATH, "r", encoding="UTF-8") as f:
    #     lines = f.readlines()

    # timestamp_regex = r"\[.{0,10}[0-9]+:[0-9]+:[0-9]+\]"

    # transcript = " ".join(
    #     [
    #         ":".join(
    #             l.split(":")[1:]
    #             if len(l.split(":")) > 1 and not l.startswith("\t")
    #             else [l]
    #         )
    #         .replace("\n", "")
    #         .replace("\t", "")
    #         .lower()
    #         for l in lines
    #     ]
    # )

    # transcript = re.sub(timestamp_regex, "", transcript)
    # transcript = re.sub(punctuation_regex, " ", transcript)
    # transcript = transcript.split()


    # bundle = torchaudio.pipelines.MMS_FA

    # emissions = calculate_emissions("path", bundle)


    # save_spans(
    #     word_spans,
    #     labels,
    #     emissions.size(1),
    #     waveform.size(1),
    #     bundle.sample_rate,
    # )


if __name__ == "__main__":
    main()
