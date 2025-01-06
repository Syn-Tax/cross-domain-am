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

TEXT_ENCODER = "FacebookAI/roberta-base"
AUDIO_ENCODER = "facebook/wav2vec2-base-960h"

MAX_TOKENS = 32
MAX_SAMPLES = 16_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-5
DROPOUT = 0.2
GRAD_ACCUMULATION_STEPS = 8

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

# initialise wandb
if "--log" in sys.argv:
    wandb.init(
        project="cross-domain-am",
        name=f"{DATA_DIR.split("/")[-1]}-{TEXT_ENCODER.split("/")[-1]}-{AUDIO_ENCODER.split("/")[-1]}-{config['merge_strategy']}-{EPOCHS}",
        config=config,
    )

# load metrics
f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")


def move_batch(batch):
    out = {}
    for key, value in batch.items():
        if key == "label":
            out[key] = value.to(device)

        else:
            out[key] = {k: v.to(device) for k, v in value.items()}

    return out


def metrics_fn(logits, targets, step="eval"):
    preds = torch.argmax(logits, dim=-1)
    macro_f1_score = f1.compute(predictions=preds, references=targets, labels=[0, 1, 2, 3], average="macro")["f1"]
    micro_f1_score = f1.compute(predictions=preds, references=targets, labels=[0, 1, 2, 3], average="micro")["f1"]
    class_f1_score = f1.compute(predictions=preds, references=targets, labels=[0, 1, 2, 3], average=None)["f1"]
    accuracy_score = accuracy.compute(predictions=preds, references=targets)["accuracy"]
    precision_score = precision.compute(predictions=preds, references=targets, average="macro")["precision"]
    recall_score = recall.compute(predictions=preds, references=targets, average="macro")["recall"]

    res = {
        f"{step}/macro_f1": macro_f1_score,
        f"{step}/micro_f1": micro_f1_score,
        f"{step}/NO_f1": float(class_f1_score[0]),
        f"{step}/RA_f1": float(class_f1_score[1]),
        f"{step}/CA_f1": float(class_f1_score[2]),
        f"{step}/MA_f1": float(class_f1_score[3]),
        f"{step}/accuracy": accuracy_score,
        f"{step}/macro_precision": precision_score,
        f"{step}/macro_recall": recall_score
    }
    print(res)

    if "--log" in sys.argv:
        wandb.log(res)


def train_step(batch, index, model, loss_fn, optim, lr_scheduler, last_batch=False):
    # batch = {k: v.to(device) for k, v in batch.items()}
    batch = move_batch(batch)
    logits = model(**batch)

    loss = loss_fn(logits, batch["label"]) / GRAD_ACCUMULATION_STEPS
    loss.backward()

    if "--log" in sys.argv:
        wandb.log({"train/train_loss": loss, "train/lr": torch.tensor(lr_scheduler.get_last_lr()[0])})

    # print(logits)
    # print(batch["label"])
    # print(loss)
    # print(lr_scheduler.get_last_lr())

    # pre_params = model.parameters()

    if index % GRAD_ACCUMULATION_STEPS == 0 or last_batch:
        optim.step()
        lr_scheduler.step()
        optim.zero_grad()

    return logits.to(torch.device("cpu")), batch["label"].to(torch.device("cpu"))


def eval(test_dataloader, model, metrics):
    progress_bar = tqdm.auto.tqdm(range(len(test_dataloader)))
    logits = torch.tensor([], dtype=torch.float, device=torch.device("cpu"))
    targets = torch.tensor([], dtype=torch.int, device=torch.device("cpu"))

    for i, batch in enumerate(test_dataloader):
        # batch = {k: v.to(device) for k, v in batch.items()}
        batch = move_batch(batch)

        with torch.no_grad():
            raw_logits = model(**batch)

        batch_logits = raw_logits.to(torch.device("cpu"))
        batch_targets = batch["label"].to(torch.device("cpu"))
        logits = torch.cat((logits, batch_logits), dim=0)
        targets = torch.cat((targets, batch_targets), dim=0)

        progress_bar.update(1)

    metrics(logits, targets, step="eval")


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
        [max(train_dataset.weights.values()) / v for k, v in train_dataset.weights.items()], device=device
    )

    print(class_weights)

    model = ConcatModel(
        TEXT_ENCODER,
        AUDIO_ENCODER,
        text_config.hidden_size,
        audio_config.hidden_size,
        dropout=DROPOUT,
    )

    # model = transformers.AutoModelForSequenceClassification.from_pretrained(TEXT_ENCODER, num_labels=4)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, warm_up_steps=0)
    lr_scheduler = transformers.get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(EPOCHS * len(train_dataloader)),
    )


    for epoch in range(EPOCHS):
        print(f"############# EPOCH {epoch} #############")

        model.train()
        # logits, targets = train_step(batch, 0, model, loss_fn, optimizer, lr_scheduler)

        # training loop
        logits = torch.tensor([], dtype=torch.float, device=torch.device("cpu"))
        targets = torch.tensor([], dtype=torch.int, device=torch.device("cpu"))

        progress_bar = tqdm.auto.tqdm(range(len(train_dataloader)))
        for i, batch in enumerate(train_dataloader):
            batch_logits, batch_targets = train_step(
                batch,
                i,
                model,
                loss_fn,
                optimizer,
                lr_scheduler,
                last_batch=(i == len(train_dataloader) - 1)
            )

            logits = torch.cat((logits, batch_logits), dim=0)
            targets = torch.cat((targets, batch_targets), dim=0)

            progress_bar.update(1)

        # get training metrics
        metrics_fn(logits, targets, step="train")

        # evaluate model
        model.eval()
        eval(
            test_dataloader,
            model,
            metrics_fn
        )

    # save model
    name = f"{DATA_DIR.split("/")[-1]}-{TEXT_ENCODER.split("/")[-1]}-{AUDIO_ENCODER.split("/")[-1]}-{config['merge_strategy']}-{EPOCHS}"
    torch.save(model.state_dict(), f"saves/{name}.pt")

    # finish wandb run
    if "--log" in sys.argv:
        wandb.save(f"saves/{name}.pt")
        wandb.finish()


if __name__ == "__main__":
    main()
