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
import time
import accelerate

from create_datasets import *
from models import *
from eval import metrics_fn, id_eval, cd_eval, load_cd
from utils import move_batch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

accelerator = accelerate.Accelerator()
device = accelerator.device


# data parameters
ID_DATA_DIR = "data/Question Time"
CD_DIRS = [
    "data/Moral Maze/Banking",
    "data/Moral Maze/Empire",
    "data/Moral Maze/Families",
    "data/Moral Maze/GreenBelt",
    "data/Moral Maze/Hypocrisy",
    "data/Moral Maze/Money",
    "data/Moral Maze/Syria",
    "data/Moral Maze/Welfare",
]
QT_COMPLETE = True

# model parameters
TEXT_ENCODER = "FacebookAI/roberta-base"
AUDIO_ENCODER = "facebook/wav2vec2-base-960h"

dataset_type = AudioOnlyDatasetConcat
model_type = AudioOnlyEarlyModel

MAX_TOKENS = 128
MAX_SAMPLES = 320_000

HEAD_HIDDEN_LAYERS = 2
HEAD_HIDDEN_SIZE = 256

# Training hyperparameters
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 1e-5
DROPOUT = 0.2
GRAD_ACCUMULATION_STEPS = 8

WEIGHT_DECAY = 0
GRAD_CLIP = 1

# configuration dictionary passed to wandb
config = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "lr": LEARNING_RATE,
    "data_dir": ID_DATA_DIR,
    "text": TEXT_ENCODER,
    "audio": AUDIO_ENCODER,
    "dropout": DROPOUT,
    "weight_decay": WEIGHT_DECAY,
    "head_size": HEAD_HIDDEN_SIZE,
    "head_layers": HEAD_HIDDEN_LAYERS,
    "max_tokens": MAX_TOKENS,
    "max_samples": MAX_SAMPLES,
    "model": model_type.__name__,
}

# set seeds
seed = 0
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class LossTrainer(transformers.Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.compute_loss_func(outputs, labels)

        return (loss, outputs) if return_outputs else loss


def main(
    epochs,
    batch_size,
    lr,
    l2,
    text_dropout,
    audio_dropout,
    head_size,
    head_layers,
    optim,
    activation,
    weighted_loss,
    freeze_encoders,
    initialisation,
    text_encoder_dropout,
    audio_encoder_dropout,
    grad_clip,
    log=False,
    init=True,
    file_append="",
):
    # load/generate datasets
    print("#### train ####")
    train_dataset = dataset_type.load(
        ID_DATA_DIR + f"/train{file_append}.json",
        ID_DATA_DIR,
        TEXT_ENCODER,
        AUDIO_ENCODER,
        MAX_TOKENS,
        MAX_SAMPLES,
        qt_complete=QT_COMPLETE,
    )

    print("#### eval ####")
    eval_dataset = dataset_type.load(
        ID_DATA_DIR + f"/eval{file_append}.json",
        ID_DATA_DIR,
        TEXT_ENCODER,
        AUDIO_ENCODER,
        MAX_TOKENS,
        MAX_SAMPLES,
        qt_complete=QT_COMPLETE,
    )

    # calculate class weights for use in the weighted cross entropy loss
    class_weights = [
        max(train_dataset.weights.values()) / v
        for k, v in train_dataset.weights.items()
    ]

    class_weights_t = torch.tensor(
        class_weights,
        device=device,
    )

    # class_weights_cpu = torch.tensor(class_weights, device=torch.device("cpu"))

    # load the model
    model = model_type(
        TEXT_ENCODER,
        AUDIO_ENCODER,
        head_hidden_layers=head_layers,
        head_hidden_size=head_size,
        text_dropout=text_dropout,
        audio_dropout=audio_dropout,
        text_encoder_dropout=text_encoder_dropout,
        audio_encoder_dropout=audio_encoder_dropout,
        activation=activation,
        freeze_encoders=freeze_encoders,
        initialisation=initialisation,
        n_classes=3,
    )
    # model = nn.DataParallel(model)
    # model.to(device)
    model = accelerator.prepare(model)

    # initialise wandb
    if log and init:
        wandb.init(
            project="cross-domain-am",
            name=f"{ID_DATA_DIR.split("/")[-1]}-{TEXT_ENCODER.split("/")[-1]}-{AUDIO_ENCODER.split("/")[-1]}-{str(model_type.__name__)}-{EPOCHS}",
            config=config,
        )

    # load loss function, optimiser and linear learning rate scheduler
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)
    # loss_fn = nn.CrossEntropyLoss()

    def calc_loss(outputs, targets, num_items_in_batch=None):
        return loss_fn(outputs["logits"], targets)

    get_params = filter(lambda p: p.requires_grad, model.parameters())

    if optim == "adamw":
        optimizer = torch.optim.AdamW(get_params, lr=lr, weight_decay=l2)
    elif optim == "adam":
        optimizer = torch.optim.Adam(get_params, lr=lr, weight_decay=l2)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(get_params, lr=lr, weight_decay=l2)
    elif optim == "rmsprop":
        optimizer = torch.optim.RMSprop(get_params, lr=lr, weight_decay=l2)

    training_args = transformers.TrainingArguments(
        output_dir="saves/",
        eval_strategy="epoch",
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=lr,
        weight_decay=l2,
        num_train_epochs=epochs,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        report_to="wandb" if "--log" in sys.argv else "none",
        remove_unused_columns=False,
        label_names=["labels"],
        bf16=True,
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="eval_macro_f1",
        save_strategy="epoch",
        dataloader_num_workers=0,
        torch_empty_cache_steps=10,
    )

    trainer = LossTrainer(
        model,
        training_args,
        collate_fn,
        compute_loss_func=calc_loss,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metrics_fn,
    )

    trainer.train()

    # save model
    name = f"{ID_DATA_DIR.split("/")[-1]}-{TEXT_ENCODER.split("/")[-1]}-{AUDIO_ENCODER.split("/")[-1]}-{model_type.__name__}-{EPOCHS}"
    torch.save(model.state_dict(), f"saves/{name}.pt")

    # cross-domain evaluation
    # print("#### cross domain ####")
    # cd_datasets = load_cd(
    #     CD_DIRS,
    #     TEXT_ENCODER,
    #     AUDIO_ENCODER,
    #     MAX_TOKENS,
    #     MAX_SAMPLES,
    #     dataset_type,
    # )

    # cd_eval(cd_datasets, [x.split("/")[-1] for x in CD_DIRS], trainer)

    # # finish wandb run
    if "--log" in sys.argv:
        wandb.save(f"saves/{name}.pt")
        wandb.finish()


if __name__ == "__main__":
    main(
        EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE,
        WEIGHT_DECAY,
        DROPOUT,
        DROPOUT,
        HEAD_HIDDEN_SIZE,
        HEAD_HIDDEN_LAYERS,
        "adamw",
        "gelu",
        True,
        False,
        "kaiming_normal",
        0.15,
        0,
        GRAD_CLIP,
        ("--log" in sys.argv),
        True,
        "-3-US",
    )
