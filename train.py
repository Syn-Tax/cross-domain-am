import torch
import torch.nn as nn
import pickle
from pathlib import Path
from create_datasets import MultimodalDataset, collate_fn
from model import ConcatModel
import transformers
import tqdm

DATA_DIR = "data/Moral Maze/GreenBelt"
TRAIN_SPLIT = 0.8

TEXT_ENCODER = "google-bert/bert-base-uncased"
AUDIO_ENCODER = "facebook/wav2vec2-base-960h"

MAX_TOKENS = 128
MAX_SAMPLES = 160_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 1e-5


def metrics_fn(eval_pred):
    pass


def train(train_dataloader, model, loss_fn, optim, metrics_fn):
    model.to(device)

    for epoch in range(EPOCHS):
        progress_bar = tqdm.tqdm(range(len(train_dataloader)))
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # print(batch)
            logits = model(**batch)

            loss = loss_fn(logits, batch["label"])
            loss.backward()
            optim.step()
            progress_bar.update(1)

            optim.zero_grad()
            print(loss)
            return


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
    )
    test_dataset = MultimodalDataset(
        DATA_DIR,
        TEXT_ENCODER,
        AUDIO_ENCODER,
        MAX_TOKENS,
        MAX_SAMPLES,
        train_test_split=TRAIN_SPLIT,
        train=False,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, BATCH_SIZE, collate_fn=collate_fn, shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, BATCH_SIZE, collate_fn=collate_fn, shuffle=True
    )

    text_config = transformers.PretrainedConfig.from_pretrained(TEXT_ENCODER)
    audio_config = transformers.PretrainedConfig.from_pretrained(AUDIO_ENCODER)

    model = ConcatModel(
        TEXT_ENCODER,
        AUDIO_ENCODER,
        text_config.hidden_size * MAX_TOKENS,
        audio_config.hidden_size * 499,
    )

    loss = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), LEARNING_RATE)

    train(train_dataloader, model, loss, optimiser, metrics_fn)


if __name__ == "__main__":
    main()
