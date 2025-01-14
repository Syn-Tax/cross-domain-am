import json
import torch
import torchaudio.functional as F
from tqdm import tqdm
import math

from datastructs import Segment

QT_EPISODE = "01.28May2020"

AUDIO_PATH = f"raw_data/Question Time/{QT_EPISODE}/audio.wav"
TRANSCRIPT_PATH = f"data/Question Time/{QT_EPISODE}/transcript.txt"

OUT_PATH = f"data/Question Time/{QT_EPISODE}/alignments.json"

CHUNK_LEN = 5  # chunk audio into 5 second increments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def align(emission, tokens):
    """Method to align tokens to emissions using pytorch forced align

    Args:
        emission (torch.Tensor): model emissions
        tokens (torch.Tensor): tokens to align

    Returns:
        torch.Tensor, torch.Tensor: alignments and confidence scores per timestep
    """
    targets = torch.tensor([tokens], dtype=torch.int32, device=torch.device("cpu"))
    emission = emission.to(torch.device("cpu"))
    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]
    scores = scores.exp()
    return alignments, scores


def unflatten(list_, lengths):
    """Method to unflatten 1d list into 2d by word

    Args:
        list_ (list): 1d list to unflatten
        lengths (list): list of lengths for each dim1 sequence

    Returns:
        list: unflattened list
    """
    assert len(list_) == sum(lengths)

    i = 0
    ret = []

    for l in lengths:
        ret.append(list_[i : i + l])
        i += l

    return ret


def calculate_emissions(waveform, sample_rate, bundle):
    """Method to get model emissions for long waveform

    Args:
        waveform (torch.Tensor): audio data
        sample_rate (int): audio's sample rate
        bundle (_type_): model bundle

    Returns:
        torch.Tensor: model emissions
    """

    # split waveform into chunks
    n_splits = math.ceil(waveform.size(1) / (CHUNK_LEN * sample_rate))
    waveform_split = torch.tensor_split(waveform, n_splits, dim=1)

    # load model
    model = bundle.get_model(with_star=False).to(device)

    # initialise output
    emissions = torch.tensor([]).to(device)

    # calculate emissions for each chunk and append to output
    for chunk in tqdm(waveform_split):
        with torch.inference_mode():
            emission, _ = model(chunk.to(device))
            emissions = torch.cat((emissions, emission), dim=1)

    # return emissions
    return emissions


def get_span(emissions, bundle, transcript, waveform_len, sample_rate):
    """Method to get the span for a certain partial transcript

    Args:
        emissions (torch.Tensor): model emissions
        bundle (_type_): model bundle
        transcript (str): partial transcript
        waveform_len (int): number of samples in the waveform
        sample_rate (int): waveform's sample rate

    Returns:
        Segment: the span information
    """
    # load and initialise dictionary
    dictionary = bundle.get_dict(star=None)
    dictionary["*"] = len(dictionary)

    # add dimension for star token
    star_dim = torch.zeros(
        (1, emissions.size(1), 1), device=emissions.device, dtype=emissions.dtype
    )
    emissions = torch.cat((emissions, star_dim), 2)

    # split transcript and add beginning and end star tokens
    transcript = transcript.split()
    transcript.insert(0, "*")
    transcript.append("*")

    # tokenize transcript
    tokenized_transcript = [dictionary[c] for word in transcript for c in word]

    # get word-level alignments
    aligned_tokens, alignment_scores = align(emissions, tokenized_transcript)
    token_spans = F.merge_tokens(aligned_tokens, alignment_scores)
    word_spans = unflatten(token_spans, [len(word) for word in transcript])[1:-1]

    # calculate word stand and end points (in seconds)
    ratio = waveform_len / emissions.size(1)

    transcript_start = int(word_spans[0][0].start * ratio) / sample_rate
    transcript_end = int(word_spans[-1][-1].end * ratio) / sample_rate

    # get mean confidence score for entire word
    score = sum([s.score for s in token_spans]) / len(token_spans)

    # construct and return span data
    return Segment(transcript, transcript_start, transcript_end, score)
