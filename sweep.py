import wandb
import torch
import gc

from train import main


def train(con=None):
    with wandb.init(config=con):
        con = wandb.config

        main(
            20,
            64,
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
    "metric": {"goal": "maximize", "name": "eval/eval/macro_f1"},
}

parameters = {
    "weight_decay": {"min": 1e-5, "max": 1e-1},
    "text_dropout": {"min": 0.0, "max": 0.8},
    "text_encoder_dropout": {"min": 0.0, "max": 0.2},
    "grad_clip": {"min": 0.3, "max": 2.0},
}

sweep_config["parameters"] = parameters

sweep_id = wandb.sweep(sweep_config, project="cross-domain-am")
print(sweep_id)

print("starting sweep.")
run = wandb.agent(sweep_id, train, count=25)
