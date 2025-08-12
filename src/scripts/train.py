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
    

def run_optuna_distributed(rank, world_size, storage_url, config):
    ddp_setup(rank, world_size)
    study = optuna.load_study(study_name=config["neptune_project"], storage=storage_url)
    study.optimize(Objective(config, world_size=world_size), n_trials=config["n_trials"], gc_after_trial=True)
    destroy_process_group()


def ddp_setup(rank, world_size):
        """
        Args:
            rank: Unique identifier of each process
            world_size: Total number of processes
        """
        master_addr = subprocess.check_output(
            "scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1", 
            shell=True,
            text=True
        ).strip()
        if os.environ["SYSTEMNAME"] in ["juwels", "jurecadc", "juwelsbooster", "jusuf"]:
            master_addr += "i" #adress on the inifinity band

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = "12355"
        torch.cuda.set_device(rank)
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

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

    # 2. Train model
    
    storage_path = f"{config["neptune_project"]}.db"
    storage_url = f"sqlite:///{storage_path}"
    study = optuna.create_study(direction="maximize", storage=storage_url, load_if_exists=True, study_name=config["neptune_project"])
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        mp.spawn(
            run_optuna_distributed,
            args=(world_size, storage_url, config),
            nprocs=world_size,
        )
        study = optuna.load_study(study_name=config["neptune_project"], storage=storage_url)
    else:
        study.optimize(Objective(config), n_trials=config["n_trials"], gc_after_trial=True)

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