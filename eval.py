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


def metrics_fn(predictions, step="eval"):
    """Method to calculate the metric scores for a specific set of logits and target labels

    Args:
        logits (torch.Tensor): the logit outputs of the model
        targets (torch.Tensor): the target labels - 1st dimensions of logits and targets must match
        step (str, optional): the training or eval step (used to log to wandb). Defaults to "eval".
    """
    logits = predictions.predictions
    targets = predictions.label_ids

    # calculate the predicted labels from logits
    preds = np.argmax(logits, axis=-1)

    # calculate metric scores
    macro_f1_score = f1.compute(
        predictions=preds, references=targets, labels=[0, 1, 2], average="macro"
    )["f1"]

    micro_f1_score = f1.compute(
        predictions=preds, references=targets, labels=[0, 1, 2], average="micro"
    )["f1"]

    class_f1_score = f1.compute(
        predictions=preds, references=targets, labels=[0, 1, 2], average=None
    )["f1"]

    accuracy_score = accuracy.compute(predictions=preds, references=targets)["accuracy"]

    precision_score = precision.compute(
        predictions=preds, references=targets, average="macro"
    )["precision"]

    recall_score = recall.compute(
        predictions=preds, references=targets, average="macro"
    )["recall"]

    # loss = loss_fn(logits, targets)
    class_names = ["None", "Support", "Attack"]

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
        f"micro_f1": micro_f1_score,
        f"NO_f1": float(class_f1_score[0]),
        f"RA_f1": float(class_f1_score[1]),
        f"CA_f1": float(class_f1_score[2]),
        # f"MA_f1": float(class_f1_score[3]),
        f"accuracy": accuracy_score,
        f"macro_precision": precision_score,
        f"macro_recall": recall_score,
        # f"{step}/epoch_loss": float(loss),
    }

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
    qt_complete=False,
):
    datasets = []

    for dir in data_dirs:
        dataset = dataset_type.load(
            dir + "/complete.json",
            dir,
            text_encoder,
            audio_encoder,
            max_tokens,
            max_samples,
            qt_complete=qt_complete,
        )

        datasets.append(dataset)

    return datasets
