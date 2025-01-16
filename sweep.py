import wandb
import torch
import gc

from train import main


def train(con=None):
    with wandb.init(config=con):
        con = wandb.config

        main(
            con["batch_size"],
            con["lr"],
            con["weight_decay"],
            con["text_dropout"],
            con["audio_dropout"],
            con["head_size"],
            con["head_layers"],
            con["optimizer"],
            con["activation"],
            con["weighted_loss"],
            con["freeze_encoders"],
            con["initialisation"],
            log=True,
            init=False,
        )

    gc.collect()
    torch.cuda.empty_cache()


sweep_config = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "eval/ID/macro_f1"},
}

parameters = {
    "batch_size": {"values": [1, 2, 4, 8, 16]},
    "lr": {"min": 1e-6, "max": 1e-5},
    "text_dropout": {"min": 0.0, "max": 0.8},
    "audio_dropout": {"min": 0.0, "max": 0.8},
    "weight_decay": {"min": 1e-7, "max": 1e-3},
    "head_size": {"values": [32, 64, 128, 256]},
    "head_layers": {"values": [0, 1, 2, 4, 8]},
    "optimizer": {"values": ["adamw", "adam", "sgd", "rmsprop"]},
    "activation": {"values": ["relu", "gelu", "tanh", "sigmoid"]},
    "weighted_loss": {"values": [True, False]},
    "freeze_encoders": {"values": [True, False]},
    "initialisation": {
        "values": [
            "kaiming_normal",
            "kaiming_uniform",
            "uniform",
            "normal",
            "xavier_uniform",
            "xavier_normal",
            None,
        ]
    },
}

sweep_config["parameters"] = parameters

sweep_id = wandb.sweep(sweep_config, project="cross-domain-am")
print(sweep_id)

print("starting sweep.")
run = wandb.agent(sweep_id, train, count=50)
