import json
import ctc_forced_aligner as fa
import torch

AUDIO_PATH = "raw_data/Moral Maze/GreenBelt/audio.mp3"
TRANSCRIPT_PATH = "raw_data/Moral Maze/GreenBelt/transcript.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
language = "eng"
batch_size = 1


def align():
    model, tokenizer = fa.load_alignment_model(
        device, dtype=torch.float16 if device == "cuda" else torch.float32
    )

    waveform = fa.load_audio(AUDIO_PATH, model.dtype, model.device)

    with open(TRANSCRIPT_PATH, "r") as f:
        lines = f.readlines()

    text = "".join(line for line in lines).replace("\n", " ").strip()

    emissions, stride = fa.generate_emissions(model, waveform, batch_size=batch_size)

    tokens_starred, text_starred = fa.preprocess_text(
        text, romanize=True, language=language
    )

    segments, scores, blank_id = fa.get_alignments(emissions, tokens_starred, tokenizer)

    spans = fa.get_spans(tokens_starred, segments, tokenizer.decode(blank_id))

    word_timestamps = fa.postprocess_results(text_starred, spans, stride, scores)

    print(word_timestamps)


if __name__ == "__main__":
    align()
