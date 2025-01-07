import pickle
import itertools
import torch
import torchaudio
import transformers
from pathlib import Path
from tqdm import tqdm
import random

from datastructs import Node, Sample


RELATION_TYPES = {"NO": 0, "RA": 1, "CA": 2, "MA": 3}
TRAIN_SPLIT = 1


class MultimodalDataset(torch.utils.data.Dataset):
    """Dataset containing multimodal sequence pairs
    """

    def __init__(
        self,
        data_dir,
        tokenizer,
        feature_extractor,
        max_tokens,
        max_samples,
        train_test_split=1,
        train=True,
        qt_complete=False
    ):
        """Class constructor

        Args:
            data_dir (str): path to data directory
            tokenizer (str): text model checkpoint
            feature_extractor (str): audio model checkpoint
            max_tokens (int): maximum length to which to pad/truncate text sequences
            max_samples (int): maximum length to pad/truncate audio sequences
            train_test_split (float, optional): proportion of data to be put in the training split. Defaults to 1.
            train (bool, optional): whether the requested split is the training split. Defaults to True.
            qt_complete (bool, optional): whether the dataset is the complete QT30 set. Defaults to False.
        """

        self.data_dir = data_dir

        self.max_tokens = max_tokens
        self.max_samples = max_samples

        self.qt_complete = qt_complete

        # load tokenizer and feature extractor
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        self.feature_extractor = transformers.AutoProcessor.from_pretrained(
            feature_extractor
        )

        # load argument map
        with open(Path(data_dir) / "argument_map.json", "r") as f:
            self.argument_map = Node.schema().loads(f.read(), many=True)

        # loop through all combinations of nodes (there must be a more efficient way to do this but it's only a minute or so on QT30)
        self.sequence_pairs = []
        relation_sequence_pairs = []
        no_relation_sequence_pairs = []
        counter = 0
        num_ra = 0
        for n1, n2 in tqdm(itertools.combinations(self.argument_map, 2)):
            # ignore combinations where neither node has a relation
            if n1.relations == [] and n2.relations == []:
                continue
            
            # check for the label type
            label = 0
            if n2.id in [r.to_node_id for r in n1.relations]:
                idx = [r.to_node_id for r in n1.relations].index(n2.id)
                label = RELATION_TYPES[n1.relations[idx].type]

            if n1.id in [r.to_node_id for r in n2.relations]:
                idx = [r.to_node_id for r in n2.relations].index(n1.id)
                label = RELATION_TYPES[n2.relations[idx].type]

            # add to counter if RA relationh
            if label == RELATION_TYPES["RA"]:
                num_ra += 1

            # if there is no relation, add to separate list - this is used to sample the NO samples
            if label == 0:
                if self.qt_complete and n1.episode != n2.episode: continue # we ignore nodes which are not in the same QT episode
                if counter % 10 == 0: # only add every 10th sample to save memory
                    no_relation_sequence_pairs.append(
                        Sample(n1, n2, torch.tensor([label], dtype=torch.long))
                    )
                counter += 1
            else:
                relation_sequence_pairs.append(
                    Sample(n1, n2, torch.tensor([label], dtype=torch.long))
                )

        # add node pairs with relations
        self.sequence_pairs.extend(relation_sequence_pairs)

        # sample node pairs without relations for NO label
        self.sequence_pairs.extend(random.sample(no_relation_sequence_pairs, num_ra))

        # shuffle dataset
        random.shuffle(self.sequence_pairs)

        # split into training and testing splits
        if train:
            self.sequence_pairs = self.sequence_pairs[
                : int(train_test_split * len(self.sequence_pairs))
            ]
        else:
            self.sequence_pairs = self.sequence_pairs[
                int(train_test_split * len(self.sequence_pairs)) :
            ]

        # calculate and display some dataset metrics
        print("------------ DATASET DATA -------------")
        print(f"length: {len(self)}")

        self.weights = {
            x: round(
                [p.label for p in self.sequence_pairs].count(x)
                / len(self.sequence_pairs),
                2,
            )
            for x in [0, 1, 2, 3]
        }
        self.counts = {
            x: [p.label for p in self.sequence_pairs].count(x) for x in [0, 1, 2, 3]
        }
        print(self.weights)
        print(self.counts)

    def __len__(self):
        """Method to get the length of the dataset
        """
        return len(self.sequence_pairs)

    def __getitem__(self, idx):
        """Method to get the item at a specific index

        Args:
            idx (int): index

        Returns:
            dict: sample
        """

        # get the relevant pair
        sample = self.sequence_pairs[idx]

        # load the audio data
        if self.qt_complete:
            n1_audio_path = str(Path(self.data_dir) / sample.node_1.episode / "audio" / (str(sample.node_1.id) + ".wav"))
            n2_audio_path = str(Path(self.data_dir) / sample.node_2.episode / "audio" / (str(sample.node_2.id) + ".wav"))
        else:
            n1_audio_path = str(Path(self.data_dir) / "audio" / (str(sample.node_1.id) + ".wav"))
            n2_audio_path = str(Path(self.data_dir) / "audio" / (str(sample.node_2.id) + ".wav"))

        n1_audio, rate = torchaudio.load(n1_audio_path)
        self.sample_rate = rate
        n2_audio, _ = torchaudio.load(n2_audio_path)

        # tokenize the text sequences
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

        # process the audio sequences
        audio1 = self.feature_extractor(
            n1_audio[0],
            max_length=self.max_samples,
            sampling_rate=rate,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True
        )
        audio2 = self.feature_extractor(
            n2_audio[0],
            max_length=self.max_samples,
            sampling_rate=rate,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True
        )

        # return the sample
        return {
            "audio1": audio1,
            "text1": text1,
            "audio2": audio2,
            "text2": text2,
            "label": sample.label,
        }


def collate_fn(data):
    """Method to collate a series of samples into a batch

    Args:
        data (list): list of samples

    Returns:
        dict: batch
    """
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
    # testing code

    TEXT_ENCODER = "google-bert/bert-base-uncased"
    AUDIO_ENCODER = "facebook/wav2vec2-base"

    MAX_TOKENS = 128
    MAX_SAMPLES = 160_000


    train_dataset = MultimodalDataset(
        "data/Moral Maze/Hypocrisy", TEXT_ENCODER, AUDIO_ENCODER, MAX_TOKENS, MAX_SAMPLES, train_test_split=TRAIN_SPLIT, train=True, qt_complete=False
    )
    test_dataset = MultimodalDataset(
        "data/Moral Maze/Hypocrisy", TEXT_ENCODER, AUDIO_ENCODER, MAX_TOKENS, MAX_SAMPLES, train_test_split=TRAIN_SPLIT, train=False, qt_complete=True
    )
