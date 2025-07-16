from google import genai
from google.genai import types
import time
from tqdm import tqdm
import wandb
from pathlib import Path
import random

from create_datasets import LLMDataset
from eval import metrics_fn

data_dirs = [
    # ("data/Question Time", True),
    ("data/Moral Maze/Banking", False),
    # ("data/Moral Maze/Empire", False),
    # ("data/Moral Maze/Families", False),
    # ("data/Moral Maze/GreenBelt", False),
    # ("data/Moral Maze/Hypocrisy", False),
    # ("data/Moral Maze/Money", False),
    # ("data/Moral Maze/Syria", False),
    # ("data/Moral Maze/Welfare", False),
]
model = "gemini-2.0-flash-lite"

client = genai.Client(api_key="AIzaSyAddBcQM2w-SYS6DOp8wJwEI7o-RbCD9Rs")

prompt = """
You are a 3 class classifier, predicting the relationship between pairs of sentences in a debate. You will classify them as one of support, attack or unrelated using only those specific words and no others.
The text proposition is immediately followed by the audio proposition. OUTPUT ONLY ONE OF 'support', 'attack' OR 'unrelated'.

The sequences are related by a support if one sentence is used to provide a reason to accept another sentence. The sequences are related by an attack if one proposition is used to provide an incompatible alternative to the other. If neither hold, the pair is unrelated. Three examples of each class are provided.

"""

relation_types = {"None": 0, "Support": 1, "Attack": 2}


# choose examples
qt30 = LLMDataset.load(
    Path("data/Question Time/train-3-SCS.json"),
    "data/Question Time",
    relation_types,
    True
)

all_no_relation = []
all_supports = []
all_attacks = []

for sample in qt30:
    if sample["labels"] == 0:
        all_no_relation.append(sample)
    elif sample["labels"] == 1:
        all_supports.append(sample)
    elif sample["labels"] == 2:
        all_attacks.append(sample)

no_samples = random.sample(all_no_relation, 3)
support_samples = random.sample(all_supports, 3)
attack_samples = random.sample(all_attacks, 3)


def get_examples(samples):
    output = []
    for s in samples:
        with open(s["audio1"], "rb") as f:
            audio1 = f.read()

        with open(s["audio2"], "rb") as f:
            audio2 = f.read()

        output.extend([
            s["text1"],
            types.Part.from_bytes(
                data=audio1,
                mime_type="audio/wav"
            ),
            s["text2"],
            types.Part.from_bytes(
                data=audio2,
                mime_type="audio/wav"
            )
        ])
    return output


examples = [
    "What follows are three examples of unrelated pairs.\n",
    *get_examples(no_samples),
    "\nWhat follows are three examples of support pairs.\n",
    *get_examples(support_samples),
    "\nWhat follows are three examples of attack pairs.\n",
    *get_examples(attack_samples),
    "\n\nThe next sequences are those on which a prediction should be made.\n"
]

for data_dir, qt_complete in data_dirs:
    dataset = LLMDataset.load(
        Path(data_dir) / ("test-3-SCS.json" if qt_complete else "complete-3-SCS.json"),
        data_dir,
        relation_types,
        qt_complete=qt_complete,
    )

    wandb.init(
        project="cross-domain-am",
        name=f"{model}-MULTI-{data_dir.split("/")[-1]}-3shot"
    )

    preds = []
    targets = []

    for sample in tqdm(dataset):
        with open(sample["audio1"], "rb") as f:
            audio1 = f.read()

        with open(sample["audio2"], "rb") as f:
            audio2 = f.read()

        while True:
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash-lite",
                    contents=[
                        prompt,
                        *examples,
                        sample["text1"],
                        types.Part.from_bytes(
                            data=audio1,
                            mime_type="audio/wav"
                        ),
                        sample["text2"],
                        types.Part.from_bytes(
                            data=audio2,
                            mime_type="audio/wav"
                        )
                    ]
                )
                break
            except Exception as e:
                print(e)
                time.sleep(10)
                continue

        labels = [1 if x in response.text.lower() else 0 for x in ["unrelated", "support", "attack"]]
        if sum(labels) != 1:
            output = -1
        else:
            output = labels.index(1)

        preds.append(output)
        targets.append(sample["labels"])

    wandb.log(metrics_fn(preds, targets))
    wandb.finish()
