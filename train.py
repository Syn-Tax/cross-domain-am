import torch
import torch.nn as nn
import random
import os
import numpy as np
import transformers
import evaluate
import tqdm
import wandb
import sys

from create_datasets import MultimodalDataset, collate_fn
from models.concat import ConcatLateModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# data parameters
DATA_DIR = "data/Moral Maze/GreenBelt"
QT_COMPLETE = False
TRAIN_SPLIT = 0.8

# model parameters
TEXT_ENCODER = "FacebookAI/roberta-large"
AUDIO_ENCODER = "facebook/wav2vec2-large-960h"

MAX_TOKENS = 32
MAX_SAMPLES = 16_000


# Training hyperparameters
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-5
DROPOUT = 0.1
GRAD_ACCUMULATION_STEPS = 8

# configuration dictionary passed to wandb
config = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "lr": LEARNING_RATE,
    "data_dir": DATA_DIR,
    "text": TEXT_ENCODER,
    "audio": AUDIO_ENCODER,
    "dropout": DROPOUT,
    "merge_strategy": "concatenation",
}

# set seeds
seed = 0
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# load metrics
f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")


def move_batch(batch):
    """Method to move a batch to the required device

    This is required to deal with the nested dict batch structure

    Args:
        batch (dict): the minibatch of tensors

    Returns:
        dict: the same minibatch but all tensors are on the required device
    """
    out = {}
    for key, value in batch.items():
        if key == "label":
            out[key] = value.to(device)

        else:
            out[key] = {k: v.to(device) for k, v in value.items()}

    return out


def metrics_fn(logits, targets, step="eval"):
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
    }

    # print out to the logs
    print(res)

    # log metrics to wandb
    if "--log" in sys.argv:
        wandb.log(res)


def train_step(batch, index, model, loss_fn, optim, lr_scheduler, last_batch=False):
    """Method to complete one training step

    Args:
        batch (dict): minibatch to be trained on
        index (int): index of the current batch in the dataloader
        model (torch.nn.Module): the model to be trained
        loss_fn (torch.nn.Module): loss/cost function
        optim (torch.optimizer): the model's optimiser
        lr_scheduler (_type_): learning rate scheduler
        last_batch (bool, optional): whether this is the last batch of an epoch. Defaults to False.

    Returns:
        torch.Tensor: the logit and target tensors on the CPU to allow training metric calculation
    """

    # move the batch to the required device
    batch = move_batch(batch)

    # get model outputs
    logits = model(**batch)

    # calculate loss and perform backward pass
    loss = loss_fn(logits, batch["label"]) / GRAD_ACCUMULATION_STEPS
    loss.backward()

    # log loss and learning rate to wandb
    if "--log" in sys.argv:
        wandb.log(
            {
                "train/train_loss": loss,
                "train/lr": torch.tensor(lr_scheduler.get_last_lr()[0]),
            }
        )

    # after the gradient accumulation steps, update the model parameters
    if index % GRAD_ACCUMULATION_STEPS == 0 or last_batch:
        optim.step()
        lr_scheduler.step()
        optim.zero_grad()

    # return the logit and target tensors
    return logits.to(torch.device("cpu")), batch["label"].to(torch.device("cpu"))


def eval(test_dataloader, model, metrics):
    """Method to evaluate model performance on a certain test dataset

    Args:
        test_dataloader (torch.utils.data.DataLoader): the test dataloader on which to evaluate
        model (torch.nn.Module): the model to evaluate
        metrics (function): the metric calculation method
    """

    # initialise some variables - importantly logits and targets to store complete arrays
    progress_bar = tqdm.auto.tqdm(range(len(test_dataloader)))
    logits = torch.tensor([], dtype=torch.float, device=torch.device("cpu"))
    targets = torch.tensor([], dtype=torch.int, device=torch.device("cpu"))

    # loop through each batch in the test dataloader
    for i, batch in enumerate(test_dataloader):
        # move the batch to the required device
        batch = move_batch(batch)

        # without calculating gradients, get the model outputsh
        with torch.no_grad():
            raw_logits = model(**batch)

        # move logits and targets back to the CPU and add them to the total tensors
        batch_logits = raw_logits.to(torch.device("cpu"))
        batch_targets = batch["label"].to(torch.device("cpu"))
        logits = torch.cat((logits, batch_logits), dim=0)
        targets = torch.cat((targets, batch_targets), dim=0)

        progress_bar.update(1)

    # calculate and log the metrics
    metrics(logits, targets, step="eval")


def main():
    # load/generate datasets
    train_dataset = MultimodalDataset(
        DATA_DIR,
        TEXT_ENCODER,
        AUDIO_ENCODER,
        MAX_TOKENS,
        MAX_SAMPLES,
        train_test_split=TRAIN_SPLIT,
        train=True,
        qt_complete=QT_COMPLETE,
    )
    test_dataset = MultimodalDataset(
        DATA_DIR,
        TEXT_ENCODER,
        AUDIO_ENCODER,
        MAX_TOKENS,
        MAX_SAMPLES,
        train_test_split=TRAIN_SPLIT,
        train=False,
        qt_complete=QT_COMPLETE,
    )

    # create dataloaders for each dataset - batching and shuffling each set
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, BATCH_SIZE, collate_fn=collate_fn, shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, collate_fn=collate_fn, shuffle=True
    )

    # calculate class weights for use in the weighted cross entropy loss
    class_weights = torch.tensor(
        [
            max(train_dataset.weights.values()) / v
            for k, v in train_dataset.weights.items()
        ],
        device=device,
    )

    # load the model
    model = ConcatLateModel(
        TEXT_ENCODER,
        AUDIO_ENCODER,
        dropout=DROPOUT,
    )
    model.to(device)
    # initialise wandb
    if "--log" in sys.argv:
        wandb.init(
            project="cross-domain-am",
            name=f"{DATA_DIR.split("/")[-1]}-{TEXT_ENCODER.split("/")[-1]}-{AUDIO_ENCODER.split("/")[-1]}-{config['merge_strategy']}-{EPOCHS}",
            config=config,
        )

    # load loss function, optimiser and linear learning rate scheduler
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = transformers.get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(EPOCHS * len(train_dataloader)),
    )

    # epoch loop
    for epoch in range(EPOCHS):
        print(f"############# EPOCH {epoch} #############")

        model.train()

        ############ training loop

        # create empty tensors for epoch's logits and targets to calculate training metrics
        logits = torch.tensor([], dtype=torch.float, device=torch.device("cpu"))
        targets = torch.tensor([], dtype=torch.int, device=torch.device("cpu"))

        # loop through each batch in the training dataloader and perform a training step
        progress_bar = tqdm.auto.tqdm(range(len(train_dataloader)))
        for i, batch in enumerate(train_dataloader):
            batch_logits, batch_targets = train_step(
                batch,
                i,
                model,
                loss_fn,
                optimizer,
                lr_scheduler,
                last_batch=(i == len(train_dataloader) - 1),
            )

            logits = torch.cat((logits, batch_logits), dim=0)
            targets = torch.cat((targets, batch_targets), dim=0)

            progress_bar.update(1)

        # get training metrics
        metrics_fn(logits, targets, step="train")

        # evaluate model
        model.eval()
        eval(test_dataloader, model, metrics_fn)

    # save model
    name = f"{DATA_DIR.split("/")[-1]}-{TEXT_ENCODER.split("/")[-1]}-{AUDIO_ENCODER.split("/")[-1]}-{config['merge_strategy']}-{EPOCHS}"
    torch.save(model.state_dict(), f"saves/{name}.pt")

    # finish wandb run
    if "--log" in sys.argv:
        wandb.save(f"saves/{name}.pt")
        wandb.finish()


if __name__ == "__main__":
    main()
