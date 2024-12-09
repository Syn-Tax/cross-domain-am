import pickle
import itertools
import torch
import torchaudio
import transformers
from pathlib import Path
from tqdm import tqdm

from datastructs import Node, Sample


RELATION_TYPES = {"RA": 1, "CA": 2, "MA": 3}
TRAIN_SPLIT = 0.8


class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        tokenizer,
        feature_extractor,
        max_tokens,
        max_samples,
        train_test_split=1,
        train=True,
    ):

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        self.feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
            feature_extractor
        )
        self.data_dir = data_dir

        self.max_tokens = max_tokens
        self.max_samples = max_samples

        with open(Path(data_dir) / "argument_map.json", "r") as f:
            self.argument_map = Node.schema().loads(f.read(), many=True)

        self.sequence_pairs = []
        for n1, n2 in tqdm(itertools.combinations(self.argument_map, 2)):
            if n1.relations == [] and n2.relations == []:
                continue

            label = 0
            if n2.id in [r.to_node_id for r in n1.relations]:
                idx = [r.to_node_id for r in n1.relations].index(n2.id)
                label = RELATION_TYPES[n1.relations[idx].type]

            if n1.id in [r.to_node_id for r in n2.relations]:
                idx = [r.to_node_id for r in n2.relations].index(n1.id)
                label = RELATION_TYPES[n2.relations[idx].type]

            if label == 0:
                continue

            self.sequence_pairs.append(
                Sample(n1, n2, torch.tensor([label - 1], dtype=torch.long))
            )

        if train:
            self.sequence_pairs = self.sequence_pairs[
                : int(train_test_split * len(self.sequence_pairs))
            ]
        else:
            self.sequence_pairs = self.sequence_pairs[
                int(train_test_split * len(self.sequence_pairs)) :
            ]

        print("------------ DATASET DATA -------------")
        print(f"length: {len(self)}")
        self.weights = {
            x: round(
                [p.label for p in self.sequence_pairs].count(x)
                / len(self.sequence_pairs),
                2,
            )
            for x in [0, 1, 2]
        }

    def __len__(self):
        return len(self.sequence_pairs)

    def __getitem__(self, idx):
        sample = self.sequence_pairs[idx]
        n1_audio_path = Path(self.data_dir) / "audio" / (str(sample.node_1.id) + ".wav")
        n2_audio_path = Path(self.data_dir) / "audio" / (str(sample.node_2.id) + ".wav")

        n1_audio, rate = torchaudio.load(n1_audio_path)
        self.sample_rate = rate
        n2_audio, _ = torchaudio.load(n2_audio_path)

        text1 = self.tokenizer(
            sample.node_1.proposition,
            max_length=self.max_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        text2 = self.tokenizer(
            sample.node_2.proposition,
            max_length=self.max_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        audio1 = self.feature_extractor(
            n1_audio[0],
            max_length=self.max_samples,
            sampling_rate=rate,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        audio2 = self.feature_extractor(
            n2_audio[0],
            max_length=self.max_samples,
            sampling_rate=rate,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "audio1": audio1,
            "text1": text1,
            "audio2": audio2,
            "text2": text2,
            "label": sample.label,
        }


def collate_fn(data):
    output = data[0]

    for sample in data[1:]:
        for root_k in data[0].keys():
            if root_k != "label":
                for k in data[0][root_k].keys():
                    output[root_k][k] = torch.cat(
                        (output[root_k][k], sample[root_k][k]), dim=0
                    )

            else:
                output[root_k] = torch.cat((output[root_k], sample[root_k]), dim=0)

    return output


if __name__ == "__main__":
    train_dataset = MultimodalDataset(
        "data/Moral Maze/GreenBelt", train_test_split=TRAIN_SPLIT, train=True
    )
    test_dataset = MultimodalDataset(
        "data/Moral Maze/GreenBelt", train_test_split=TRAIN_SPLIT, train=False
    )

    with open("data/Moral Maze/GreenBelt/train_dataset.pkl", "wb+") as f:
        pickle.dump(train_dataset, f)

    with open("data/Moral Maze/GreenBelt/test_dataset.pkl", "wb+") as f:
        pickle.dump(test_dataset, f)
