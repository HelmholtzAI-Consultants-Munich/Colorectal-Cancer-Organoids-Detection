import argparse
import os
import shutil
from time import time
import subprocess

import gdown
import yaml
import torch
from torch.utils.data import DataLoader
import optuna
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp



from src.model.model import maskRCNNModel, maskRCNNModelFreeze
from src.model.dataset import MaskRCNNDataset
from src.model.objective import Objective

def get_args():
    parser = argparse.ArgumentParser(description="Train a model to detect organoids in images.")
    parser.add_argument("-c","--config", type=str, help="Path to the configuration file containing the hyperparameters.")

    return parser.parse_args()
    

def worker(rank, n_trials, storage_url, config, device):
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(rank)

    study = optuna.load_study(study_name=config["neptune_project"], storage=storage_url)
    study.optimize(Objective(config, device=device), n_trials=n_trials, gc_after_trial=True)



def main():
    ### 1. Get args and initialize model + datasets (val and train)
    # args
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  
    args = get_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # create model folder
    config["model_dir"] = os.path.join(config["model_dir"], str(int(time())))
    os.makedirs(config["model_dir"], exist_ok=True)

    # set random seed
    torch.manual_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(config["seed"])

    # 2. Train model
    
    storage_path = f"{config["neptune_project"]}.db"
    storage_url = f"sqlite:///{storage_path}"
    study = optuna.create_study(direction="maximize", storage=storage_url, load_if_exists=True, study_name=config["neptune_project"])
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        print(f"Number of GPUs: {world_size}")
        # end the programm
        mp.spawn(
            worker,
            args=(config["n_trials"]//world_size, storage_url, config, device),
            nprocs=world_size,
        )
        study = optuna.load_study(study_name=config["neptune_project"], storage=storage_url)
    else:
        print("No GPUs available")
        study.optimize(Objective(config, device=device), n_trials=config["n_trials"], gc_after_trial=True)

    # 3. Save best trial
    print(f"Best trial: {study.best_trial.value}")
    shutil.copy(
        os.path.join(config["model_dir"], f"best-checkpoint-{study.best_trial.number}.bin"), 
        os.path.join(config["model_dir"], "best_model.pth")
    )

    # delete storage, remove if at next iteration you want to resume 
    os.remove(storage_path) 

    pass

if __name__ == '__main__':
    main()