import itertools
import torch
import torchaudio
import transformers
from pathlib import Path
from tqdm import tqdm
import random

from datastructs import Node, Sample


RELATION_TYPES = {"NO": 0, "RA": 1, "CA": 2, "MA": 3}
SPLITS = [0.7, 0.1, 0.2]

AUDIO_EOS_LEN = 2.5

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def process(data_dir, qt_complete, splits):
    # load argument map
    with open(Path(data_dir) / "argument_map.json", "r") as f:
        argument_map = Node.schema().loads(f.read(), many=True)

    # loop through all combinations of nodes (there must be a more efficient way to do this but it's only a minute or so on QT30)
    sequence_pairs = []
    relation_sequence_pairs = []
    no_relation_sequence_pairs = []
    counter = 0
    num_ra = 0
    for n1, n2 in tqdm(itertools.combinations(argument_map, 2)):
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
            if qt_complete and n1.episode != n2.episode:
                continue  # we ignore nodes which are not in the same QT episode
            if counter % 10 == 0:  # only add every 10th sample to save memory
                no_relation_sequence_pairs.append(Sample(n1, n2, label))
            counter += 1
        else:
            relation_sequence_pairs.append(Sample(n1, n2, label))

    # add node pairs with relations
    sequence_pairs.extend(relation_sequence_pairs)

    # sample node pairs without relations for NO label
    sequence_pairs.extend(random.sample(no_relation_sequence_pairs, num_ra))

    # shuffle dataset
    random.shuffle(sequence_pairs)

    # get requested split
    split_data = []
    for s in range(len(splits)):
        start = sum(splits[:s])
        end = start + splits[s]

        print(start)
        print(end)

        if end == 0:
            end = 1

        split_data.append(
            sequence_pairs[
                int(start * len(sequence_pairs)) : int(end * len(sequence_pairs))
            ]
        )

    return split_data


def get_metrics(data):
    # calculate and display some dataset metrics
    print("------------ DATASET DATA -------------")
    print(f"length: {len(data)}")

    weights = {
        x: round(
            [p.labels for p in data].count(x) / len(data),
            2,
        )
        for x in [0, 1, 2, 3]
    }
    counts = {x: [p.labels for p in data].count(x) for x in [0, 1, 2, 3]}
    print(weights)
    print(counts)

    return weights, counts


def save(path, data):
    with open(path, "w") as f:
        out = Sample.schema().dumps(data, many=True)
        f.write(out)


class MultimodalDatasetConcat(torch.utils.data.Dataset):
    """Dataset containing multimodal sequence pairs"""

    def __init__(
        self,
        data_dir,
        tokenizer,
        feature_extractor,
        max_tokens,
        max_samples,
        train_test_split=[1, 0, 0],
        split=0,
        qt_complete=False,
        process=True,
    ):
        """Class constructor

        Args:
            data_dir (str): path to data directory
            tokenizer (str): text model checkpoint
            feature_extractor (str): audio model checkpoint
            max_tokens (int): maximum length to which to pad/truncate text sequences
            max_samples (int): maximum length to pad/truncate audio sequences
            train_test_split (list[float], optional): proportion of data to be put into each split. Defaults to entirely training split.
            split (int, optional): the requested split. Defaults to 0.
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

        if process:
            self.sequence_pairs = process(data_dir, train_test_split)[split]

    def __len__(self):
        """Method to get the length of the dataset"""
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
            n1_audio_path = str(
                Path(self.data_dir)
                / sample.node_1.episode
                / "audio"
                / (str(sample.node_1.id) + ".wav")
            )
            n2_audio_path = str(
                Path(self.data_dir)
                / sample.node_2.episode
                / "audio"
                / (str(sample.node_2.id) + ".wav")
            )
        else:
            n1_audio_path = str(
                Path(self.data_dir) / "audio" / (str(sample.node_1.id) + ".wav")
            )
            n2_audio_path = str(
                Path(self.data_dir) / "audio" / (str(sample.node_2.id) + ".wav")
            )

        n1_audio, rate = torchaudio.load(n1_audio_path)
        self.sample_rate = rate
        n2_audio, _ = torchaudio.load(n2_audio_path)

        text_proposition = (
            f"{sample.node_1.proposition} </s> {sample.node_2.proposition}"
        )

        audio_cat = torch.tensor([0 for _ in range(int(rate * AUDIO_EOS_LEN))])

        audio_proposition = torch.cat((n1_audio[0], audio_cat, n2_audio[0]))

        # tokenize the text sequences
        text = self.tokenizer(
            text_proposition,
            max_length=self.max_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # process the audio sequences
        audio = self.feature_extractor(
            audio_proposition,
            max_length=self.max_samples,
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
            "labels": torch.tensor([sample.labels], dtype=torch.long),
        }

    def save(self, path):
        with open(path, "w") as f:
            out = Sample.schema().dumps(self.sequence_pairs, many=True)
            f.write(out)

    def load(
        path,
        data_dir,
        tokenizer,
        feature_extractor,
        max_tokens,
        max_samples,
        qt_complete=False,
    ):
        s = MultimodalDatasetConcat(
            data_dir,
            tokenizer,
            feature_extractor,
            max_tokens,
            max_samples,
            qt_complete=qt_complete,
            process=False,
        )

        with open(path, "r") as f:
            s.sequence_pairs = Sample.schema().loads(f.read(), many=True)

        s.weights, s.counts = get_metrics(s.sequence_pairs)

        return s


class MultimodalDatasetNoConcat(torch.utils.data.Dataset):
    """Dataset containing multimodal sequence pairs"""

    def __init__(
        self,
        data_dir,
        tokenizer,
        feature_extractor,
        max_tokens,
        max_samples,
        train_test_split=[1, 0, 0],
        split=0,
        qt_complete=False,
        process=True,
    ):
        """Class constructor

        Args:
            data_dir (str): path to data directory
            tokenizer (str): text model checkpoint
            feature_extractor (str): audio model checkpoint
            max_tokens (int): maximum length to which to pad/truncate text sequences
            max_samples (int): maximum length to pad/truncate audio sequences
            train_test_split (list[float], optional): proportion of data to be put into each split. Defaults to entirely training split.
            split (int, optional): the requested split. Defaults to 0.
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

        if process:
            self.sequence_pairs = process(data_dir, train_test_split)[split]

    def __len__(self):
        """Method to get the length of the dataset"""
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
            n1_audio_path = str(
                Path(self.data_dir)
                / sample.node_1.episode
                / "audio"
                / (str(sample.node_1.id) + ".wav")
            )
            n2_audio_path = str(
                Path(self.data_dir)
                / sample.node_2.episode
                / "audio"
                / (str(sample.node_2.id) + ".wav")
            )
        else:
            n1_audio_path = str(
                Path(self.data_dir) / "audio" / (str(sample.node_1.id) + ".wav")
            )
            n2_audio_path = str(
                Path(self.data_dir) / "audio" / (str(sample.node_2.id) + ".wav")
            )

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
            return_attention_mask=True,
        )
        audio2 = self.feature_extractor(
            n2_audio[0],
            max_length=self.max_samples,
            sampling_rate=rate,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )

        # return the sample
        return {
            "text1": text1,
            "audio1": audio1,
            "text2": text2,
            "audio2": audio2,
            "labels": torch.tensor([sample.labels], dtype=torch.long),
        }

    def save(self, path):
        with open(path, "w") as f:
            out = Sample.schema().dumps(self.sequence_pairs, many=True)
            f.write(out)

    def load(
        path,
        data_dir,
        tokenizer,
        feature_extractor,
        max_tokens,
        max_samples,
        qt_complete=False,
    ):
        s = MultimodalDatasetNoConcat(
            data_dir,
            tokenizer,
            feature_extractor,
            max_tokens,
            max_samples,
            qt_complete=qt_complete,
            process=False,
        )

        with open(path, "r") as f:
            s.sequence_pairs = Sample.schema().loads(f.read(), many=True)

        s.weights, s.counts = get_metrics(s.sequence_pairs)

        return s


class TextOnlyDatasetConcat(torch.utils.data.Dataset):
    """Dataset containing multimodal sequence pairs"""

    def __init__(
        self,
        data_dir,
        tokenizer,
        feature_extractor,
        max_tokens,
        max_samples,
        train_test_split=[1, 0, 0],
        split=0,
        qt_complete=False,
        process=True,
    ):
        """Class constructor

        Args:
            data_dir (str): path to data directory
            tokenizer (str): text model checkpoint
            feature_extractor (str): audio model checkpoint
            max_tokens (int): maximum length to which to pad/truncate text sequences
            max_samples (int): maximum length to pad/truncate audio sequences
            train_test_split (list[float], optional): proportion of data to be put into each split. Defaults to entirely training split.
            split (int, optional): the requested split. Defaults to 0.
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

        if process:
            self.sequence_pairs = process(data_dir, train_test_split)[split]

    def __len__(self):
        """Method to get the length of the dataset"""
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

        # tokenize the text sequences
        text = f"{sample.node_1.proposition} </s> {sample.node_2.proposition}"

        text1 = self.tokenizer(
            text,
            max_length=self.max_tokens * 2,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # return the sample
        return {"labels": torch.tensor([sample.labels], dtype=torch.long), **text1}

    def save(self, path):
        with open(path, "w") as f:
            out = Sample.schema().dumps(self.sequence_pairs, many=True)
            f.write(out)

    def load(
        path,
        data_dir,
        tokenizer,
        feature_extractor,
        max_tokens,
        max_samples,
        qt_complete=False,
    ):
        s = TextOnlyDatasetConcat(
            data_dir,
            tokenizer,
            feature_extractor,
            max_tokens,
            max_samples,
            qt_complete=qt_complete,
            process=False,
        )

        with open(path, "r") as f:
            s.sequence_pairs = Sample.schema().loads(f.read(), many=True)

        s.weights, s.counts = get_metrics(s.sequence_pairs)

        return s


def collate_fn(data):
    """Method to collate a series of samples into a batch

    Args:
        data (list): list of samples

    Returns:
        dict: batch
    """

    # print(data)
    output = data[0]

    for sample in data[1:]:
        for root_k in data[0].keys():
            if root_k != "labels":
                for k in data[0][root_k].keys():
                    output[root_k][k] = torch.cat(
                        (output[root_k][k], sample[root_k][k]), dim=0
                    )

            else:
                output[root_k] = torch.cat((output[root_k], sample[root_k]), dim=0)

    return output


def collate_fn_raw(data):
    output = data[0]

    for sample in data[1:]:
        for k in data[0].keys():
            output[k] = torch.cat((output[k], sample[k]), dim=0)

    return output


if __name__ == "__main__":
    # testing code

    TEXT_ENCODER = "google-bert/bert-base-uncased"
    AUDIO_ENCODER = "facebook/wav2vec2-base"

    MAX_TOKENS = 32
    MAX_SAMPLES = 16_000

    data_dirs = [
        "data/Question Time",
        "data/Moral Maze/Banking",
        "data/Moral Maze/DDay",
        "data/Moral Maze/Empire",
        "data/Moral Maze/Families",
        "data/Moral Maze/GreenBelt",
        "data/Moral Maze/Hypocrisy",
        "data/Moral Maze/Money",
        "data/Moral Maze/Syria",
        "data/Moral Maze/Welfare",
    ]

    qt_complete = [True, False, False, False, False, False, False, False, False, False]

    for i in range(len(data_dirs)):
        print(f"############### {data_dirs[i].split('/')[-1]} ############")
        splits = process(data_dirs[i], qt_complete[i], SPLITS)
        get_metrics(splits[0])
        get_metrics(splits[1])
        get_metrics(splits[2])

        save(data_dirs[i] + "/train.json", splits[0])
        save(data_dirs[i] + "/eval.json", splits[1])
        save(data_dirs[i] + "/test.json", splits[2])

        complete = splits[0]
        complete.extend(splits[1])
        complete.extend(splits[2])

        # MultimodalDataset.save(data_dirs[i] + "/complete.json", complete)

        print("############## COMPLETE #############")
        get_metrics(complete)
