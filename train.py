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
import gc

from create_datasets import *
from models import *
from eval import metrics_fn, id_eval, cd_eval, load_cd, N_CLASSES
from utils import move_batch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

accelerator = accelerate.Accelerator()
device = accelerator.device


# data parameters
ID_DATA_DIR = "data/Moral Maze/Welfare"
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

DATASET_TYPE = MultimodalDatasetNoConcat
MODEL_TYPE = AudioOnlyLateModel

MAX_TOKENS = 64
MAX_SAMPLES = 16_000

HEAD_HIDDEN_LAYERS = 2
HEAD_HIDDEN_SIZE = 256

# Training hyperparameters
BATCH_SIZE = 1
EPOCHS = 15
LEARNING_RATE = 1e-3
DROPOUT = 0.2
GRAD_ACCUMULATION_STEPS = 1

WEIGHT_DECAY = 0
GRAD_CLIP = 1

RELATION_TYPES = {
    3: {"None": 0, "Support": 1, "Attack": 2},
    4: {"NO": 0, "RA": 1, "CA": 2, "MA": 3},
}


class LossTrainer(transformers.Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.compute_loss_func(outputs, labels)

        return (loss, outputs) if return_outputs else loss


def main(
    log=False,
    init=True,
    train_set="",
    eval_set="",
    test_set="",
    cd_sets="",
    text_encoder="FacebookAI/roberta-base",
    audio_encoder="facebook/wav2vec2-base-960h",
    dataset_type=None,
    model_type=None,
    mm_fusion_method="concat",
    n_classes=4,
):
    # set seeds
    seed = 0
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    N_CLASSES = n_classes

    # load/generate datasets
    print("#### train ####")
    train_dataset = dataset_type.load(
        ID_DATA_DIR + f"/{train_set}.json",
        ID_DATA_DIR,
        text_encoder,
        audio_encoder,
        MAX_TOKENS,
        MAX_SAMPLES,
        RELATION_TYPES[n_classes],
        qt_complete=QT_COMPLETE,
    )

    print("#### eval ####")
    eval_dataset = dataset_type.load(
        ID_DATA_DIR + f"/{eval_set}.json",
        ID_DATA_DIR,
        text_encoder,
        audio_encoder,
        MAX_TOKENS,
        MAX_SAMPLES,
        RELATION_TYPES[n_classes],
        qt_complete=QT_COMPLETE,
    )

    print("#### test ####")
    test_dataset = dataset_type.load(
        ID_DATA_DIR + f"/{test_set}.json",
        ID_DATA_DIR,
        text_encoder,
        audio_encoder,
        MAX_TOKENS,
        MAX_SAMPLES,
        RELATION_TYPES[n_classes],
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
        head_hidden_layers=HEAD_HIDDEN_LAYERS,
        head_hidden_size=HEAD_HIDDEN_SIZE,
        text_dropout=DROPOUT,
        audio_dropout=DROPOUT,
        n_classes=4,
        mm_fusion_method=mm_fusion_method,
    )

    model = accelerator.prepare(model)

    # configuration dictionary passed to wandb
    config = {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "data_dir": ID_DATA_DIR,
        "text": text_encoder,
        "audio": audio_encoder,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "head_size": HEAD_HIDDEN_SIZE,
        "head_layers": HEAD_HIDDEN_LAYERS,
        "max_tokens": MAX_TOKENS,
        "max_samples": MAX_SAMPLES,
        "model": model_type.__name__,
        "dataset": dataset_type.__name__,
        "train_set": train_set,
        "eval_set": eval_set,
        "test_set": test_set,
        "mm_fusion_method": mm_fusion_method,
    }

    # initialise wandb
    if log and init:
        t = mm_fusion_method.upper()
        m = f"{text_encoder.split("/")[-1]}-{audio_encoder.split("/")[-1]}"
        if "AudioOnly" in model_type.__name__:
            t = "AUDIO"
            m = audio_encoder.split("/")[-1]
        elif "TextOnly" in model_type.__name__:
            t = "TEXT"
            m = text_encoder.split("/")[-1]

        wandb.init(
            project="cross-domain-am",
            name=f"QT-{m}-{t}-LATE-{EPOCHS}-{train_set.split("-")[-2]}-4class-OS_CA",
            config=config,
        )

    # load loss function, optimiser and linear learning rate scheduler
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)
    # loss_fn = nn.CrossEntropyLoss()

    def calc_loss(outputs, targets, num_items_in_batch=None):
        return loss_fn(outputs["logits"], targets)

    training_args = transformers.TrainingArguments(
        output_dir="saves/",
        eval_strategy="epoch",
        logging_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=EPOCHS,
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

    # in-domain evaluation
    print("#### in domain ####")
    trainer.predict(test_dataset)

    # cross-domain evaluation
    print("#### cross domain ####")
    cd_datasets = load_cd(
        CD_DIRS,
        TEXT_ENCODER,
        AUDIO_ENCODER,
        MAX_TOKENS,
        MAX_SAMPLES,
        dataset_type,
        cd_sets,
        RELATION_TYPES[n_classes],
    )

    cd_eval(cd_datasets, [x.split("/")[-1] for x in CD_DIRS], trainer)

    # # finish wandb run
    if "--log" in sys.argv:
        wandb.save(f"saves/{name}.pt")
        wandb.finish()

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main(
        log=("--log" in sys.argv),
        train_set="train-4-SCS",
        eval_set="eval-4-SCS",
        test_set="test-4-SCS",
        cd_sets="complete-4-SCS",
        dataset_type=DATASET_TYPE,
        model_type=MODEL_TYPE,
        n_classes=4,
    )
