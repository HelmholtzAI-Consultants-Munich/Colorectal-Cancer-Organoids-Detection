from typing import Dict
import os

from optuna import Trial
import torch

from src.model.model import *
from src.model.dataset import MaskRCNNDataset
from src.model.engine import FitterMaskRCNN
from src.utils.utils import load_pretrained_weights


class Objective:

    def __init__(self, config: Dict, device):
        self.config = config
        self.device = device


    def initialize_model(self, hyperparameters: dict) -> torch.nn.Module:
        #model
        if self.config["model_name"] == "mask_rcnn":
            if hyperparameters["freeze_weights"]:
                model = maskRCNNModelFreeze()
            else:
                model = maskRCNNModel() # TODO: Freeze the weights if we want to train under these conditions
            model_weights = load_pretrained_weights(self.device)
            model.load_state_dict(model_weights)
            model.to(self.device)
        elif self.config["model_name"] == "faster_rcnn":
            model = fasterRCNNModel(backbone=hyperparameters["backbone"], trainable_layers=hyperparameters["trainable_layers"])
            model.to(self.device)
        return model


    def __call__(self, trial: Trial):

        fitter = FitterMaskRCNN(device=self.device)
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
        if "freeze_weights" in self.config.keys():
            hyperparams["freeze_weights"] = trial.suggest_categorical(
                name="freeze_weights",
                choices=self.config["freeze_weights"],
            )
        if "backbone" in self.config.keys():
            hyperparams["backbone"] = trial.suggest_categorical(
                name="backbone",
                choices=self.config["backbone"],
            )
        if "trainable_layers" in self.config.keys():
            hyperparams["trainable_layers"] = trial.suggest_categorical(
                name="trainable_layers",
                choices=self.config["trainable_layers"],
            )
        hyperparams["data_augmentation"] = trial.suggest_categorical(
            name="data_augmentation",
            choices=self.config["data_augmentation"],
        )
        hyperparams["n_epochs"] = self.config["n_epochs"]
        hyperparams["patience"] = self.config["patience"]
        if hyperparams["batch_size"] > 8:
            hyperparams["accumulation_steps"] = hyperparams["batch_size"] // 8
        else:
            hyperparams["accumulation_steps"] = 1

        # Initialize the dataset
        if "annotator" in self.config:
            annotator = self.config["annotator"]
        else:
            annotator = None

        hyperparams["project"] = self.config["neptune_project"]

            

        train_dataset = MaskRCNNDataset(self.config["train_dataset_path"], datatype="train", data_augmentation=hyperparams["data_augmentation"], annotator=annotator)
        val_dataset = MaskRCNNDataset(self.config["val_dataset_path"], datatype="eval", annotator=annotator)

        # test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=self.collate_fn)
        # initialize the model
        model = self.initialize_model(hyperparams)

        # Train the model
        try:

            val_mpa = fitter.fit( 
                trial.number,
                model,
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