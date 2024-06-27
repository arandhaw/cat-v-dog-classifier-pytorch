#!/usr/bin/env python
# coding: utf-8
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import os
import subprocess
import logging
import tempfile

import ray
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
import matplotlib.pyplot as plt
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

output_dir = "/home/arand/cat-v-dog-classifier-pytorch/results"

# function to run bash commands
# print determines whether output is printed
def bash(command, show_result = False):
    try:
        if show_result == True:
            print("Running:", command)
        result = subprocess.run(command.split(), capture_output=True, text=True, check=True)
        if show_result:
            print(result.stdout)
    except Exception as e:
        print("Bash command failed:", command)
        print("Error message\n", e)

# loads data on remote machine and returns directory
def load_data():
    if not os.path.exists("/cat_dog/training_data"):
        bash("sudo mkdir /cat_dog")
        bash("sudo chmod ugo+rwx /cat_dog")
        bash("sudo gsutil cp -r gs://yakoa-model-data/Cats_and_dogs/* /cat_dog")
        bash("sudo unzip /cat_dog/training_data.zip -d /cat_dog", show_result = False)
    if not os.path.exists("/cat_dog/training_data"):
        print("ERROR!!!!! The data directory doesn't exist")
    return "/cat_dog/training_data"


def training_function():
    print("hello world")
    time.sleep(2)
    load_data()
    time.sleep(2)
    print(f"Shared directory status: {os.path.exists('/mnt/shared')}")
    time.sleep(2)
    print("Is torch available", torch.cuda.is_available())
    

trainer = ray.train.torch.TorchTrainer(
    training_function,
    scaling_config = ray.train.ScalingConfig(num_workers=2, use_gpu=True),
    # [5a] If running in a multi-node cluster, this is where you
    # should configure the run's persistent storage that is accessible
    # across all worker nodes.
)

result = trainer.fit()
print("Training complete!")