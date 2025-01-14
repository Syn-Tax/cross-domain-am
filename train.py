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
from eval import metrics_fn, id_eval, cd_eval, load_cd
from utils import move_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# data parameters
ID_DATA_DIR = "data/Moral Maze/GreenBelt"
CD_DIRS = [
    "data/Moral Maze/Banking",
    "data/Moral Maze/Empire",
    "data/Moral Maze/Families",
    # "data/Moral Maze/GreenBelt",
    "data/Moral Maze/Hypocrisy",
    "data/Moral Maze/Money",
    "data/Moral Maze/Syria",
    "data/Moral Maze/Welfare",
]
QT_COMPLETE = False

# model parameters
TEXT_ENCODER = "FacebookAI/roberta-base"
AUDIO_ENCODER = "facebook/wav2vec2-base-960h"

MAX_TOKENS = 32
MAX_SAMPLES = 16_000


# Training hyperparameters
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-5
DROPOUT = 0
GRAD_ACCUMULATION_STEPS = 8

# configuration dictionary passed to wandb
config = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "lr": LEARNING_RATE,
    "data_dir": ID_DATA_DIR,
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
    batch = move_batch(batch, device)

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


def main():
    # load/generate datasets
    print("#### train ####")
    train_dataset = MultimodalDataset.load(
        ID_DATA_DIR + "/train.json",
        ID_DATA_DIR,
        TEXT_ENCODER,
        AUDIO_ENCODER,
        MAX_TOKENS,
        MAX_SAMPLES,
        qt_complete=QT_COMPLETE,
    )

    print("#### eval ####")
    eval_dataset = MultimodalDataset.load(
        ID_DATA_DIR + "/eval.json",
        ID_DATA_DIR,
        TEXT_ENCODER,
        AUDIO_ENCODER,
        MAX_TOKENS,
        MAX_SAMPLES,
        qt_complete=QT_COMPLETE,
    )

    # create dataloaders for each dataset - batching and shuffling each set
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, BATCH_SIZE, collate_fn=collate_fn, shuffle=True
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, BATCH_SIZE, collate_fn=collate_fn, shuffle=True
    )

    # load cross domain evaluation sets
    print("#### cross domain ####")
    cd_dataloaders = load_cd(
        CD_DIRS,
        BATCH_SIZE,
        collate_fn,
        TEXT_ENCODER,
        AUDIO_ENCODER,
        MAX_TOKENS,
        MAX_SAMPLES,
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
            name=f"{ID_DATA_DIR.split("/")[-1]}-{TEXT_ENCODER.split("/")[-1]}-{AUDIO_ENCODER.split("/")[-1]}-{config['merge_strategy']}-{EPOCHS}",
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
        id_eval(eval_dataloader, model, metrics_fn, device)

    # perform cross domain evaluation
    model.eval()
    cd_eval(
        cd_dataloaders, [d.split("/")[-1] for d in CD_DIRS], model, metrics_fn, device
    )

    # save model
    name = f"{ID_DATA_DIR.split("/")[-1]}-{TEXT_ENCODER.split("/")[-1]}-{AUDIO_ENCODER.split("/")[-1]}-{config['merge_strategy']}-{EPOCHS}"
    torch.save(model.state_dict(), f"saves/{name}.pt")

    # finish wandb run
    if "--log" in sys.argv:
        wandb.save(f"saves/{name}.pt")
        wandb.finish()


if __name__ == "__main__":
    main()
