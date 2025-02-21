import torch
import transformers
import sys

import train
from models import *
from create_datasets import *

dataset_configs = [
    ("train-4-SCS", "eval-4-SCS", "test-4-SCS"),
    ("train-4-LCS", "eval-4-SCS", "test-4-SCS"),
    ("train-4-US", "eval-4-SCS", "test-4-SCS"),
]

model_dataset_types = [
    (TextOnlyDatasetConcat, TextOnlyEarlyModel),
    (AudioOnlyDatasetConcat, AudioOnlyEarlyModel),
    (MultimodalDatasetConcat, MultimodalEarlyLateModel),
    (MultimodalDatasetNoConcat, MultimodalLateLateModel),
]

fusion_params = {
    "MultimodalEarlyLateModel": ["concat", "prod", "ca_text", "ca_audio"]
}

for train_set, eval_set, test_set in dataset_configs:
    for dataset_type, model_type in model_dataset_types:
        if model_type.__name__ in fusion_params.keys():
            for fusion_param in fusion_params[model_type.__name__]:
                train.main(
                    log="--log" in sys.argv,
                    init=True,
                    train_dataset=train_set,
                    eval_dataset=eval_set,
                    test_dataset=test_set,
                    dataset_type=dataset_type,
                    model_type=model_type,
                    mm_fusion_method=fusion_param,
                    n_classes=4
                )

        else:
            train.main(
                log="--log" in sys.argv,
                init=True,
                train_dataset=train_set,
                eval_dataset=eval_set,
                test_dataset=test_set,
                dataset_type=dataset_type,
                model_type=model_type,
                n_classes=4
            )