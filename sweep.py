import wandb
import torch
import gc

from train import main


def train(con=None):
    with wandb.init(config=con):
        con = wandb.config

        main(
            con["epochs"],
            16,
            2e-5,
            con["weight_decay"],
            con["text_dropout"],
            0.0,
            0,
            0,
            "adamw",
            "gelu",
            False,
            True,
            None,
            con["text_encoder_dropout"],
            0.0,
            con["grad_clip"],
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
    "epochs": {"min": 1, "max": 50},
    "weight_decay": {"min": 1e-7, "max": 1e-3},
    "text_dropout": {"min": 0.0, "max": 0.8},
    "text_encoder_dropout": {"min": 0.0, "max": 0.3},
    "grad_clip": {"min": 0.3, "max": 2.0},
}

sweep_config["parameters"] = parameters

sweep_id = wandb.sweep(sweep_config, project="cross-domain-am")
print(sweep_id)

print("starting sweep.")
run = wandb.agent(sweep_id, train, count=25)
