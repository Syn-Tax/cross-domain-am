import itertools
import torch
import torchaudio
import transformers
from pathlib import Path
from tqdm import tqdm
import random
import math
import sys

from datastructs import Node, Sample

SPLITS = [0.7, 0.1, 0.2]
AUDIO_EOS_LEN = 5
audio_cat = torch.tensor([0 for _ in range(int(16_000 * AUDIO_EOS_LEN))])

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def generate_pairs(data_dir, qt_complete, splits, relation_types):
    # load argument map
    with open(Path(data_dir) / "argument_map.json", "r") as f:
        argument_map = Node.schema().loads(f.read(), many=True)

    # loop through all combinations of nodes (there must be a more efficient way to do this but it's only a minute or so on QT30)
    output = {
        "SCS": [],
        "LCS": [],
        "US": []
    }
    relation_sequence_pairs = []
    no_relation_sequence_pairs = {
        "SCS": [],
        "LCS": [],
        "US": []
    }

    counter = 0
    num_ra = 0
    for n1, n2 in tqdm(itertools.combinations(argument_map, 2)):
        # ignore combinations where neither node has a relation
        # if n1.relations == [] and n2.relations == []:
        #     continue

        # check for the label type
        label = 0
        if n2.id in [r.to_node_id for r in n1.relations]:
            idx = [r.to_node_id for r in n1.relations].index(n2.id)
            label = relation_types[n1.relations[idx].type]

        if n1.id in [r.to_node_id for r in n2.relations]:
            idx = [r.to_node_id for r in n2.relations].index(n1.id)
            label = relation_types[n2.relations[idx].type]

        # add to counter if RA relation
        if label == relation_types["RA"]:
            num_ra += 1

        # if there is no relation, add to separate list depending on sampling type - this is used to sample the NO samples
        if label == 0:
            if counter % 20 == 0:  # only add every 10th sample to save memory
                if qt_complete and n1.episode != n2.episode:
                    no_relation_sequence_pairs["LCS"].append(Sample(n1, n2, label))
                if qt_complete and n1.episode == n2.episode:
                    no_relation_sequence_pairs["SCS"].append(Sample(n1, n2, label))

                if not qt_complete:
                    no_relation_sequence_pairs["LCS"].append(Sample(n1, n2, label))
                    no_relation_sequence_pairs["SCS"].append(Sample(n1, n2, label))

                no_relation_sequence_pairs["US"].append(Sample(n1, n2, label))

            counter += 1
        else:
            relation_sequence_pairs.append(Sample(n1, n2, label))

    # add node pairs with relations
    related_splits = split_data(relation_sequence_pairs, splits)
    output["SCS"].extend([s.copy() for s in related_splits])
    output["LCS"].extend([s.copy() for s in related_splits])
    output["US"].extend([s.copy() for s in related_splits])

    # add node pairs without relations
    unrelated_splits = {
        "SCS": split_data(no_relation_sequence_pairs["SCS"], splits),
        "LCS": split_data(no_relation_sequence_pairs["LCS"], splits),
        "US": split_data(no_relation_sequence_pairs["US"], splits)
    }

    # print(len(unrelated_splits["SCS"][0]), num_ra*splits[0])

    for key in output.keys():
        for i in range(len(splits)):
            x = random.sample(unrelated_splits[key][i], int(num_ra*splits[i]))
            output[key][i].extend(x)
            random.shuffle(output[key][i])

    return output


def split_data(data, splits):
    # get requested split
    split_data = []
    for s in range(len(splits)):
        start = sum(splits[:s])
        end = start + splits[s]

        if end == 0:
            end = 1

        split_data.append(
            data[
                int(start * len(data)) : int(end * len(data))
            ]
        )

    return split_data


def get_metrics(data, relation_types):
    # calculate and display some dataset metrics
    print("------------ DATASET DATA -------------")
    print(f"length: {len(data)}")

    weights = {
        x: round(
            [p.labels for p in data].count(x) / len(data),
            2,
        )
        for x in set(relation_types.values())
    }
    counts = {
        x: [p.labels for p in data].count(x) for x in set(relation_types.values())
    }
    print(weights)
    print(counts)

    return weights, counts


def save(path, data):
    with open(path, "w") as f:
        out = Sample.schema().dumps(data, many=True)
        f.write(out)


def resample(data, relation_types, sampling):
    labelled_samples = {k: [] for k in relation_types.values()}
    output = []

    for sample in data:
        labelled_samples[sample.labels].append(sample)

    sampling_floored = [math.floor(x) for x in sampling]
    sampling_dec = [x - y for x, y in zip(sampling, sampling_floored)]

    # complete integer part of resampling
    for relation, s in zip(labelled_samples.keys(), sampling_floored):
        for x in range(s):
            output.extend(labelled_samples[relation])

    # complete fractional part of resampling
    for relation, s in zip(labelled_samples.keys(), sampling_dec):
        output.extend(
            labelled_samples[relation][: int(s * len(labelled_samples[relation]))]
        )

    # reshuffle output
    random.shuffle(output)
    return output

def load(
    path,
):

    with open(path, "r") as f:
        return Sample.schema().loads(f.read(), many=True)

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
        relation_types,
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

        s.weights, s.counts = get_metrics(s.sequence_pairs, relation_types)

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

        text_proposition = (
            f"{sample.node_1.proposition} </s> {sample.node_2.proposition}"
        )

        # tokenize the text sequences
        text = self.tokenizer(
            text_proposition,
            max_length=self.max_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # return the sample
        return {
            "text": text,
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
        relation_types,
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

        s.weights, s.counts = get_metrics(s.sequence_pairs, relation_types)

        return s


class AudioOnlyDatasetConcat(torch.utils.data.Dataset):
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

        audio_proposition = torch.cat((n1_audio[0], audio_cat, n2_audio[0]))

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
        relation_types,
        qt_complete=False,
    ):
        s = AudioOnlyDatasetConcat(
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

        s.weights, s.counts = get_metrics(s.sequence_pairs, relation_types)

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
        relation_types,
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

        s.weights, s.counts = get_metrics(s.sequence_pairs, relation_types)

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

    qt_completes = [True, False, False, False, False, False, False, False, False, False]
    # qt_completes = [False, False, False, False, False, False, False, False, False]
    resamplings = {
        "OS_CA": {3: [1, 1, 3], 4: [1, 1, 3, 1]},
    }
    classes = {
        3: {"NO": 0, "RA": 1, "CA": 2, "MA": 1},
        4: {"NO": 0, "RA": 1, "CA": 2, "MA": 3},
    }

    for data_dir, qt_complete in zip(data_dirs, qt_completes):
        for class_prob in classes.keys():

            # splits = generate_pairs(
            #     data_dir, qt_complete, SPLITS, classes[class_prob]
            # )



            for no_sampling in ["SCS", "LCS", "US"]:
                print(
                    f"############### {data_dir.split('/')[-1]}-{class_prob}-{no_sampling} ############"
                )

                # get_metrics(splits["SCS"][0], classes[class_prob])
                # get_metrics(splits["SCS"][1], classes[class_prob])
                # get_metrics(splits["SCS"][2], classes[class_prob])

                # save(data_dir + f"/train-{class_prob}-{no_sampling}.json", splits[no_sampling][0])
                # save(data_dir + f"/eval-{class_prob}-{no_sampling}.json", splits[no_sampling][1])
                # save(data_dir + f"/test-{class_prob}-{no_sampling}.json", splits[no_sampling][2])

                # complete = splits[no_sampling][0]
                # complete.extend(splits[no_sampling][1])
                # complete.extend(splits[no_sampling][2])

                # save(data_dir + f"/complete-{class_prob}-{no_sampling}.json", complete)

                train = load(data_dir + f"/train-{class_prob}-{no_sampling}.json")
                get_metrics(train, classes[class_prob])

                for resampling in resamplings.keys():
                    out = resample(
                        train,
                        classes[class_prob],
                        resamplings[resampling][class_prob],
                    )
                    get_metrics(out, classes[class_prob])
                    save(
                        data_dir
                        + f"/train-{class_prob}-{no_sampling}-{resampling}.json",
                        out,
                    )
