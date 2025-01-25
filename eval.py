import torch
import evaluate
import wandb
import tqdm
import sys
import time

from utils import move_batch
from create_datasets import MultimodalDataset

# load metrics
f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")


def metrics_fn(logits, targets, loss_fn, step="eval"):
    """Method to calculate the metric scores for a specific set of logits and target labels

    Args:
        logits (torch.Tensor): the logit outputs of the model
        targets (torch.Tensor): the target labels - 1st dimensions of logits and targets must match
        step (str, optional): the training or eval step (used to log to wandb). Defaults to "eval".
    """

    # calculate the predicted labels from logits
    preds = torch.argmax(logits, dim=-1)

    # calculate metric scores
    macro_f1_score = f1.compute(
        predictions=preds, references=targets, labels=[0, 1, 2, 3], average="macro"
    )["f1"]

    micro_f1_score = f1.compute(
        predictions=preds, references=targets, labels=[0, 1, 2, 3], average="micro"
    )["f1"]

    class_f1_score = f1.compute(
        predictions=preds, references=targets, labels=[0, 1, 2, 3], average=None
    )["f1"]

    accuracy_score = accuracy.compute(predictions=preds, references=targets)["accuracy"]

    precision_score = precision.compute(
        predictions=preds, references=targets, average="macro"
    )["precision"]

    recall_score = recall.compute(
        predictions=preds, references=targets, average="macro"
    )["recall"]

    loss = loss_fn(logits, targets)

    # add metric scores to dictionary
    res = {
        f"{step}/macro_f1": macro_f1_score,
        f"{step}/micro_f1": micro_f1_score,
        f"{step}/NO_f1": float(class_f1_score[0]),
        f"{step}/RA_f1": float(class_f1_score[1]),
        f"{step}/CA_f1": float(class_f1_score[2]),
        f"{step}/MA_f1": float(class_f1_score[3]),
        f"{step}/accuracy": accuracy_score,
        f"{step}/macro_precision": precision_score,
        f"{step}/macro_recall": recall_score,
        f"{step}/epoch_loss": float(loss),
    }

    # print out to the logs
    print(res)

    # log metrics to wandb
    if "--log" in sys.argv:
        wandb.log(res)


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


def cd_eval(dataloaders, datasets, model, metrics, loss_fn, device):
    for loader, dataset in zip(dataloaders, datasets):
        eval(loader, model, metrics, device, loss_fn, loader=dataset)


def load_cd(
    data_dirs,
    batch_size,
    collate_fn,
    text_encoder,
    audio_encoder,
    max_tokens,
    max_samples,
    qt_complete=False,
):
    dataloaders = []

    for dir in data_dirs:
        dataset = MultimodalDataset.load(
            dir + "/complete.json",
            dir,
            text_encoder,
            audio_encoder,
            max_tokens,
            max_samples,
            qt_complete=qt_complete,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size, collate_fn=collate_fn, shuffle=True
        )

        dataloaders.append(dataloader)

    return dataloaders
