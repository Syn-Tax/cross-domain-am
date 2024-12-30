import torch
import torch.nn as nn
import random
import os
import numpy as np
import pickle
from pathlib import Path
from create_datasets import MultimodalDataset, collate_fn
from model import ConcatModel
import transformers
import evaluate
import tqdm
import wandb
import sys

DATA_DIR = "data/Moral Maze/GreenBelt"
QT_COMPLETE = False
TRAIN_SPLIT = 0.8

TEXT_ENCODER = "google-bert/bert-base-uncased"
AUDIO_ENCODER = "facebook/wav2vec2-base"

MAX_TOKENS = 128
MAX_SAMPLES = 160_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-5
DROPOUT = 0.2
GRAD_ACCUMULATION_STEPS = 8

# set seeds
seed = 0
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# initialise wandb
if "--log" in sys.argv:
    wandb.init(
        project="cross-domain-am",
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
            "data_dir": DATA_DIR,
            "text": TEXT_ENCODER,
            "audio": AUDIO_ENCODER,
            "dropout": DROPOUT,
            "merge_strategy": "concatenation",
        },
    )

# load metrics
f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")


def metrics_fn(logits, targets):
    preds = torch.argmax(logits, dim=-1)
    f1_score = f1.compute(predictions=preds, references=targets, average="macro")["f1"]
    accuracy_score = accuracy.compute(predictions=preds, references=targets)["accuracy"]

    res = {"f1": f1_score, "accuracy": accuracy_score}
    print(res)

    if "--log" in sys.argv:
        wandb.log(res)


def train(train_dataloader, model, loss_fn, optim, lr_scheduler):
    progress_bar = tqdm.tqdm(range(len(train_dataloader)))
    for i, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch)

        loss = loss_fn(logits, batch["label"]) / GRAD_ACCUMULATION_STEPS

        if "--log" in sys.argv:
            wandb.log({"train_loss": loss, "lr": lr_scheduler.get_last_lr()[0]})

        # pre_params = model.parameters()
        loss.backward()

        if i + 1 % GRAD_ACCUMULATION_STEPS == 0 or i + 1 == len(train_dataloader):
            optim.step()
            optim.zero_grad()

        lr_scheduler.step()
        progress_bar.update(1)


def eval(test_dataloader, model, metrics):
    progress_bar = tqdm.tqdm(range(len(test_dataloader)))
    logits = torch.tensor([], dtype=torch.float, device=torch.device("cpu"))
    targets = torch.tensor([], dtype=torch.int, device=torch.device("cpu"))

    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            raw_logits = model(**batch)

        batch_logits = raw_logits.to(torch.device("cpu"))
        batch_targets = batch["label"].to(torch.device("cpu"))
        logits = torch.cat((logits, batch_logits), dim=0)
        targets = torch.cat((targets, batch_targets), dim=0)

        progress_bar.update(1)

    metrics(logits, targets)


def main():
    # load dataset splits
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

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, BATCH_SIZE, collate_fn=collate_fn, shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, collate_fn=collate_fn, shuffle=True
    )

    text_config = transformers.PretrainedConfig.from_pretrained(TEXT_ENCODER)
    audio_config = transformers.PretrainedConfig.from_pretrained(AUDIO_ENCODER)

    class_weights = torch.tensor(
        [1 / v for k, v in train_dataset.weights.items()], device=device
    )

    model = ConcatModel(
        TEXT_ENCODER,
        AUDIO_ENCODER,
        text_config.hidden_size,
        audio_config.hidden_size,
        dropout=DROPOUT,
    )
    model.to(device)

    loss = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

    for epoch in range(EPOCHS):
        print(f"############# EPOCH {epoch} #############")

        model.train()
        train(train_dataloader, model, loss, optimizer, lr_scheduler)

        model.eval()
        eval(test_dataloader, model, metrics_fn)


if __name__ == "__main__":
    main()
