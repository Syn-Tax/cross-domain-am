import pathlib
import torchaudio
import tqdm
import sys
import numpy as np

lengths = []
samples = []

for path in tqdm.tqdm(pathlib.Path(sys.argv[1]).rglob("**/*.wav")):
    waveform, rate = torchaudio.load(path)

    lengths.append(waveform.shape[1] / rate)
    samples.append(waveform.shape[1])

lengths = np.array(lengths)
samples = np.array(samples)

print(f"Mean Length: {np.mean(lengths)}s")
print(f"Mean Samples: {np.mean(samples)}")
print(f"Total Length: {np.sum(lengths)}s")
print(f"Q3: {np.quantile(lengths, 0.75)}s")
print(f"Q1: {np.quantile(lengths, 0.25)}s")
print(f"Upper 90%: {np.quantile(lengths, 0.9)}s")
print(f"Upper 99%: {np.quantile(lengths, 0.99)}s")
print(f"Max Length: {np.max(lengths)}s")
print(f"Max Samples: {np.max(samples)}")
