import pathlib
import torchaudio

from datastructs import Node

DATA_PATH = "data/"

lengths =[]

for path in pathlib.Path(DATA_PATH).rglob("**/argument_map.json"):
    with open(path, "r") as f:
        nodes = Node.schema().loads(f.read(), many=True)

    for node in nodes:
        lengths.append(len(node.locution.split()))

print(f"max text: {max(lengths)}")
print(f"mean text: {sum(lengths) / len(lengths)}")

audio_lengths = []

for path in pathlib.Path(DATA_PATH).rglob("**/*.wav"):
    waveform, _ = torchaudio.load(path)
    audio_lengths.append(waveform.shape[1])
    del waveform

print(f"max audio: {max(audio_lengths)}")
print(f"mean audio: {sum(audio_lengths) / len(audio_lengths)}")