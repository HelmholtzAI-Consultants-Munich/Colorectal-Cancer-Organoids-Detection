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
        self.train_dataset = MaskRCNNDataset(config["train_dataset_path"], datatype="train")
        self.val_dataset = MaskRCNNDataset(config["val_dataset_path"], datatype="eval")
        self.test_dataset = MaskRCNNDataset(config["test_dataset_path"], datatype="eval")
        self.collate_fn = lambda x: tuple(zip(*x))
        


    def __call__(self, trial: Trial):
        
        fitter = FitterMaskRCNN(id=trial.number)
        # Generate the hyperparameters
        hyperparams = {}
        hyperparams["batch_size"] = trial.suggest_int(
            name="batch_size", 
            low=self.config["batch_size"]["min"],
            high=self.config["batch_size"]["max"],
            log = True,
        )
        hyperparams["learning_rate"] = trial.suggest_float(
            name="learning_rate",
            low=self.config["learning_rate"]["min"],
            high=self.config["learning_rate"]["max"],
            log = True,
        )
        hyperparams["n_epochs"] = self.config["n_epochs"]
        hyperparams["patience"] = self.config["patience"]

        # Initialize the dataloader 
        train_loader = DataLoader(self.train_dataset, batch_size=hyperparams["batch_size"], shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader(self.val_dataset, batch_size=hyperparams["batch_size"], shuffle=False, collate_fn=self.collate_fn)
        test_loader = DataLoader(self.test_dataset, batch_size=hyperparams["batch_size"], shuffle=False, collate_fn=self.collate_fn)

        # Train the model
        val_mpa = fitter.fit( 
            train_loader, 
            val_loader, 
            hyperparams,
            self.config["model_dir"],
            self.config["neptune_workspace"],
            self.config["neptune_project"],
        )

        # return the validation metric
        return val_mpa