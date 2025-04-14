from typing import Dict
import os

from optuna import Trial
import torch
from torch.utils.data import DataLoader

from src.model.model import maskRCNNModel, maskRCNNModelFreeze
from src.model.dataset import MaskRCNNDataset
from src.model.engine import FitterMaskRCNN

class Objective:

    def __init__(self, config: Dict):
        self.config = config
    
        # Initialize the dataset
        self.collate_fn = lambda x: tuple(zip(*x))


    def __call__(self, trial: Trial):

        # Get the device
        device = "cuda" if torch.cuda.is_available() else "cpu"          
        
        fitter = FitterMaskRCNN(id=trial.number, device=device)
        # Generate the hyperparameters
        hyperparams = {}
        hyperparams["batch_size"] = trial.suggest_categorical(
            name="batch_size", 
            choices=self.config["batch_size"],
        )
        hyperparams["learning_rate"] = trial.suggest_float(
            name="learning_rate",
            low=self.config["learning_rate"]["min"],
            high=self.config["learning_rate"]["max"],
            log = True,
        )
        hyperparams["freeze_weights"] = trial.suggest_categorical(
            name="freeze_weights",
            choices=self.config["freeze_weights"],
        )
        hyperparams["data_augmentation"] = trial.suggest_categorical(
            name="data_augmentation",
            choices=self.config["data_augmentation"],
        )
        hyperparams["n_epochs"] = self.config["n_epochs"]
        hyperparams["patience"] = self.config["patience"]
        if hyperparams["batch_size"] > 8:
            hyperparams["accumulation_steps"] = hyperparams["batch_size"] // 8
            loader_batch_size = 4
        else:
            hyperparams["accumulation_steps"] = 1
            loader_batch_size = hyperparams["batch_size"]

        # Initialize the dataset
        train_dataset = MaskRCNNDataset(self.config["train_dataset_path"], datatype="train", data_augmentation=hyperparams["data_augmentation"])
        val_dataset = MaskRCNNDataset(self.config["val_dataset_path"], datatype="eval")
        test_dataset = MaskRCNNDataset(self.config["test_dataset_path"], datatype="eval")
        # Initialize the dataloader 
        train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=self.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=self.collate_fn)

        # Train the model
        try:
            val_mpa = fitter.fit( 
                train_loader, 
                val_loader, 
                hyperparams,
                self.config["model_dir"],
                self.config["neptune_workspace"],
                self.config["neptune_project"],
            )
        except torch.OutOfMemoryError:
            val_mpa = 0.0
            print("Out of memory error")
        except Exception as e:
            val_mpa = 0.0
            print("Error", e)

        # Free the GPU
        if torch.cuda.is_available():
            self.gpus[device_id] = True

        # return the validation metric
        return val_mpa