import torch
import torchaudio
import transformers
from pathlib import Path

from models.multimodal import MultimodalEarlyLateModel

tokenizer = transformers.AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
feature_extractor = transformers.AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

relation_types = {
    "NO": 0,
    "RA": 1,
    "CA": 2,
    "MA": 3
}

audio_path = Path("data/Moral Maze/GreenBelt/audio")


def process_sample(text1, audio1, text2, audio2, audio_eos_len=5, rate=16_000, max_tokens=64, max_samples=320_000):
    audio_cat = torch.tensor([0 for _ in range(int(rate * audio_eos_len))])

    text_proposition = (
        f"{text1} </s> {text2}"
    )

    audio_proposition = torch.cat((audio1[0], audio_cat, audio2[0]))

    # tokenize the text sequences
    text = tokenizer(
        text_proposition,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # process the audio sequences
    audio = feature_extractor(
        audio_proposition,
        max_length=max_samples,
        sampling_rate=rate,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=True,
    )

    # return the sample
    return {
        "text": text,
        "audio": audio,
        # "labels": torch.tensor([sample.labels], dtype=torch.long),
    }


# load model
model = MultimodalEarlyLateModel(
    "FacebookAI/roberta-base",
    "facebook/wav2vec2-base-960h",
    mm_fusion_method="concat",
    head_hidden_layers=2,
    head_hidden_size=256
)

model.load_state_dict(torch.load("demo_model.pt", weights_only=False, map_location=torch.device("cpu")))
model.eval()

print("loaded model")

# load samples
text1 = "it is the principal of the green belt or the way the policy's been applied that Shiv Malik'd say he objects to"
id_1 = 267407
audio1, _ = torchaudio.load("audio1.wav")

print("#########  TEXT 1  #########")
print(text1)

text2 = "it really comes down to the policy"
id_2 = 267412
audio2, _ = torchaudio.load("audio2.wav")

print("#########  TEXT 2  #########")
print(text2)

# process samples
model_input = process_sample(text1, audio1, text2, audio2)

print("processed samples")

# put samples through model
output = torch.nn.functional.softmax(model(**model_input)["logits"], dim=1)

print("#########  OUTPUT  #########")
print("Output is: " + str(list(relation_types.keys())[torch.argmax(output[0])]))
