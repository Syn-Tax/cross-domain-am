import json
import torch
import torchaudio
import torchaudio.functional as F
import re
from tqdm import tqdm
import math

AUDIO_PATH = "raw_data/Moral Maze/DDay/audio_16000.wav"
TRANSCRIPT_PATH = "raw_data/Moral Maze/DDay/transcript.txt"

OUT_PATH = "data/Moral Maze/DDay/alignments.json"

CHUNK_LEN = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def main():
    waveform, sample_rate = torchaudio.load(AUDIO_PATH, format="mp3")

    n_splits = math.ceil(waveform.size(1) / (CHUNK_LEN * sample_rate))
    waveform_split = torch.tensor_split(waveform, n_splits, dim=1)

    with open(TRANSCRIPT_PATH, "r", encoding="UTF-8") as f:
        lines = f.readlines()

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
            for l in lines
        ]
    )

    transcript = re.sub(timestamp_regex, "", transcript)
    transcript = re.sub(punctuation_regex, " ", transcript)
    transcript = transcript.split()

    bundle = torchaudio.pipelines.MMS_FA

    model = bundle.get_model(with_star=False).to(device)
    labels = bundle.get_labels(star=None)
    dictionary = bundle.get_dict(star=None)

    tokenized_transcript = [dictionary[c] for word in transcript for c in word]

    emissions = torch.tensor([]).to(device)

    for chunk in tqdm(waveform_split):
        with torch.inference_mode():
            emission, _ = model(chunk.to(device))
            emissions = torch.cat((emissions, emission), dim=1)

    aligned_tokens, alignment_scores = align(emissions, tokenized_transcript)
    token_spans = F.merge_tokens(aligned_tokens, alignment_scores)
    word_spans = unflatten(token_spans, [len(word) for word in transcript])

    save_spans(
        word_spans,
        labels,
        emissions.size(1),
        waveform.size(1),
        bundle.sample_rate,
    )


if __name__ == "__main__":
    main()
