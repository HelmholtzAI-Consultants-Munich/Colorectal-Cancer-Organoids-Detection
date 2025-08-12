from typing import Dict
import os

from optuna import Trial
import torch

from src.model.model import maskRCNNModel, maskRCNNModelFreeze
from src.model.dataset import MaskRCNNDataset
from src.model.engine import FitterMaskRCNN
from src.utils.utils import load_pretrained_weights


class Objective:

    def __init__(self, config: Dict, world_size: int = 1):
        self.config = config
        self.world_size = world_size
    
        # Initialize the dataset
        self.collate_fn = lambda x: tuple(zip(*x))

    
    def initialize_model(self, freeze_weights) -> torch.nn.Module:
        #model
        if freeze_weights:
            model = maskRCNNModelFreeze()
        else:
            model = maskRCNNModel() # TODO: Freeze the weights if we want to train under these conditions
        model_weights = load_pretrained_weights(self.device)
        model.load_state_dict(model_weights)
        model.to(self.device)
        return model


    def __call__(self, trial: Trial):

        # Get the device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        world_size = torch.cuda.device_count()          
        
        fitter = FitterMaskRCNN(device=device)
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
        if hyperparams["batch_size"] > 8* world_size:
            hyperparams["accumulation_steps"] = hyperparams["batch_size"] // 8
        else:
            hyperparams["accumulation_steps"] = 1

        # Initialize the dataset
        if "annotator" in self.config:
            annotator = self.config["annotator"]
        else:
            annotator = None

            

        train_dataset = MaskRCNNDataset(self.config["train_dataset_path"], datatype="train", data_augmentation=hyperparams["data_augmentation"], annotator=annotator)
        val_dataset = MaskRCNNDataset(self.config["val_dataset_path"], datatype="eval", annotator=annotator)
        test_dataset = MaskRCNNDataset(self.config["test_dataset_path"], datatype="eval")

        # test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=self.collate_fn)
        # initialize the model
        model = self.initialize_model(hyperparams["freeze_weights"])

        # Train the model
        try:

            if device == "cuda":
                fit_function = fitter.fit_distributed

            else:    
                fit_function = fitter.fit
            val_mpa = fit_function( 
                trial.number,
                train_dataset, 
                val_dataset, 
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

        # return the validation metric
        return val_mpa