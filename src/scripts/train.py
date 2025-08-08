import argparse
import os
import shutil
from time import time

import gdown
import yaml
import torch
from torch.utils.data import DataLoader
import optuna

from src.model.model import maskRCNNModel, maskRCNNModelFreeze
from src.model.dataset import MaskRCNNDataset
from src.model.objective import Objective

def get_args():
    parser = argparse.ArgumentParser(description="Train a model to detect organoids in images.")
    parser.add_argument("-c","--config", type=str, help="Path to the configuration file containing the hyperparameters.")

    return parser.parse_args()

def main():
    ### 1. Get args and initialize model + datasets (val and train)
    # args
    args = get_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # create model folder
    config["model_dir"] = os.path.join(config["model_dir"], str(int(time())))
    os.makedirs(config["model_dir"], exist_ok=True)

    # set random seed
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    # 2. Train model: (see goat/engine.py)
    study = optuna.create_study(direction="maximize")
    study.optimize(Objective(config), n_trials=config["n_trails"], gc_after_trial=True)

    # 3. Save best trial
    print(f"Best trial: {study.best_trial.value}")
    shutil.copy(
        os.path.join(config["model_dir"], f"best-checkpoint-{study.best_trial.number}.bin"), 
        os.path.join(config["model_dir"], "best_model.pth")
    )

    pass

if __name__ == '__main__':
    main()