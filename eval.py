import torch
import evaluate
import wandb
import tqdm
import sys
import time
import numpy as np
import sklearn.metrics as skm
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from utils import move_batch
from create_datasets import *

# load metrics
f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

N_CLASSES = 4


def metrics_fn(predictions, targets=None, step="eval"):
    """Method to calculate the metric scores for a specific set of logits and target labels

    Args:
        logits (torch.Tensor): the logit outputs of the model
        targets (torch.Tensor): the target labels - 1st dimensions of logits and targets must match
        step (str, optional): the training or eval step (used to log to wandb). Defaults to "eval".
    """

    if targets is None:

        logits = predictions.predictions
        targets = predictions.label_ids

        # calculate the predicted labels from logits
        preds = np.argmax(logits, axis=-1)
    else:
        preds = np.array(predictions)
        targets = np.array(targets)

    # calculate metric scores
    macro_f1_score = f1.compute(
        predictions=preds,
        references=targets,
        labels=list(range(N_CLASSES)),
        average="macro",
    )["f1"]

    weighted_f1_score = f1.compute(
        predictions=preds,
        references=targets,
        labels=list(range(N_CLASSES)),
        average="weighted",
    )["f1"]

    class_f1_score = f1.compute(
        predictions=preds,
        references=targets,
        labels=list(range(N_CLASSES)),
        average=None,
    )["f1"]

    accuracy_score = accuracy.compute(predictions=preds, references=targets)["accuracy"]

    precision_score = precision.compute(
        predictions=preds, references=targets, average="macro", zero_division=0.0
    )["precision"]

    recall_score = recall.compute(
        predictions=preds, references=targets, average="macro"
    )["recall"]

    # loss = loss_fn(logits, targets)
    if N_CLASSES == 3:
        class_names = ["None", "Support", "Attack"]
    elif N_CLASSES == 4:
        class_names = ["NO", "RA", "CA", "MA"]

    cm = skm.confusion_matrix(targets, preds)
    df = pd.DataFrame(
        cm / np.sum(cm, axis=1)[:, None],
        index=[i for i in class_names],
        columns=[i for i in class_names],
    )
    plt.figure(figsize=(12, 7))
    sn.heatmap(df, annot=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # add metric scores to dictionary
    res = {
        f"macro_f1": macro_f1_score,
        f"weighted_f1": weighted_f1_score,
        f"accuracy": accuracy_score,
        f"macro_precision": precision_score,
        f"macro_recall": recall_score,
        # f"{step}/epoch_loss": float(loss),
    }

    for name, f1_score in zip(class_names, class_f1_score):
        res[f"{name}_f1"] = float(f1_score)

    print(res)

    # log metrics to wandb
    if "--log" in sys.argv:
        try:
            wandb.log({"conf_mat": wandb.Image(plt)})
        except:
            pass

    plt.close()

    return res


def eval(eval_dataloader, model, metrics, loss_fn, device, loader="ID"):
    """Method to evaluate model performance on a certain test dataset

    Args:
        test_dataloader (torch.utils.data.DataLoader): the test dataloader on which to evaluate
        model (torch.nn.Module): the model to evaluate
        metrics (function): the metric calculation method
    """

    # initialise some variables - importantly logits and targets to store complete arrays
    progress_bar = tqdm.auto.tqdm(range(len(eval_dataloader)))
    logits = torch.tensor([], dtype=torch.float, device=torch.device("cpu"))
    targets = torch.tensor([], dtype=torch.int, device=torch.device("cpu"))

    # loop through each batch in the test dataloader
    for i, batch in enumerate(eval_dataloader):
        # move the batch to the required device
        batch = move_batch(batch, device)

        # without calculating gradients, get the model outputsh
        with torch.no_grad():
            raw_logits = model(**(batch["text1"])).logits

        # move logits and targets back to the CPU and add them to the total tensors
        batch_logits = raw_logits.to(torch.device("cpu"))
        batch_targets = batch["label"].to(torch.device("cpu"))
        logits = torch.cat((logits, batch_logits), dim=0)
        targets = torch.cat((targets, batch_targets), dim=0)

        progress_bar.update(1)

    # calculate and log the metrics
    metrics(logits, targets, loss_fn, step=f"eval/{loader}")


def id_eval(eval_dataloader, model, metrics, loss_fn, device):
    eval(eval_dataloader, model, metrics, loss_fn, device)


def cd_eval(datasets, dataset_names, trainer):
    for dataset, name in zip(datasets, dataset_names):
        print(f"#################### {name} ####################")
        trainer.predict(dataset, metric_key_prefix=f"test/{name}")


def load_cd(
    data_dirs,
    text_encoder,
    audio_encoder,
    max_tokens,
    max_samples,
    dataset_type,
    dataset_name,
    relation_types,
    qt_complete=False,
):
    datasets = []

    for dir in data_dirs:
        dataset = dataset_type.load(
            dir + f"/{dataset_name}.json",
            dir,
            text_encoder,
            audio_encoder,
            max_tokens,
            max_samples,
            relation_types,
            qt_complete=qt_complete,
        )

        datasets.append(dataset)

    return datasets


def baselines(data_files, relation_types):
    for dir, qt_complete in data_files:
        print(f"##### {dir} #####")
        dataset = TextOnlyDatasetConcat.load(
            dir,
            "/".join(dir.split('/')[:-1]),
            "FacebookAI/roberta-base",
            "facebook/wav2vec2-base-960h",
            64,
            10,
            relation_types,
            qt_complete
        )

        # random baseline
        print("### Random ###")
        preds = []
        targets = []
        for sample in dataset:
            targets.append(sample["labels"])
            preds.append(random.choice(list(relation_types.values())))

        metrics_fn(preds, targets)

        # majority baseline
        print("### Majority ###")
        preds = []
        targets = []
        for sample in dataset:
            targets.append(sample["labels"])
            preds.append(1)

        metrics_fn(preds, targets)

if __name__ == "__main__":
    N_CLASSES = 3
    baselines(
        [
            # ("data/Question Time/test-4-SCS.json", True),
            # ("data/Question Time/test-4-LCS.json", True),
            # ("data/Question Time/test-4-US.json", True),
            ("data/Moral Maze/Banking/complete-3-SCS.json", False),
            ("data/Moral Maze/Empire/complete-3-SCS.json", False),
            ("data/Moral Maze/Money/complete-3-SCS.json", False),
            ("data/Moral Maze/Families/complete-3-SCS.json", False),
            ("data/Moral Maze/Syria/complete-3-SCS.json", False),
            ("data/Moral Maze/GreenBelt/complete-3-SCS.json", False),
            ("data/Moral Maze/DDay/complete-3-SCS.json", False),
            ("data/Moral Maze/Hypocrisy/complete-3-SCS.json", False),
            ("data/Moral Maze/Welfare/complete-3-SCS.json", False),
        ],
        # {"NO": 0, "RA": 1, "CA": 2, "MA": 3}
        {"None": 0, "Support": 1, "Attack": 2}
    )