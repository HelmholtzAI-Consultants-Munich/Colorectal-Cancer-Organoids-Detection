import os
from datetime import datetime
import time
from copy import deepcopy

import torch
from torch import nn, tensor
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import gdown
import neptune
from neptune.management import create_project, get_project_list
from tqdm import tqdm
import gc

from src.model.model import maskRCNNModel, maskRCNNModelFreeze

class FitterMaskRCNN():

    def __init__(self, id: str):
        # Set the device between GPU, MPS, or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id = id

    def fit(self, train_loader, val_loader, hyperparams, model_dir, neptune_workspace, neptune_project):
        # Initialize the model
        model = self.initialize_model(hyperparams["freeze_weights"])
        # Initialize optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True, factor=0.5)
        # Set up the neptune experiment
        if f"{neptune_workspace}/{neptune_project}" not in get_project_list():
            create_project(name=neptune_project, workspace=neptune_workspace)
        run = neptune.init_run(
            project=f"{neptune_workspace}/{neptune_project}", 
        )
        run.assign({"hyperparameters": hyperparams})
        run["job_id"] = os.path.split(model_dir)[-1]
        run["local_id"] = self.id
        print(f"Starting training {self.id}")

        # compute the original mAP
        val_metric = self.evaluate_one_epoch(model, val_loader)

        print(f"Original model: Validation mAP: {val_metric['map']}")
        
        # Iterate through epochs
        patience = 0
        best_map = 0.
        best_epoch = None
        best_checkpoint = deepcopy(model.state_dict())
        for epoch in range(hyperparams["n_epochs"]):
            # Train one epoch
            losses = self.train_one_epoch(model, train_loader, optimizer, accumulation_steps=hyperparams["accumulation_steps"])
            self.log_metric(losses, "training_losses", run)
            # Validate
            val_metric = self.evaluate_one_epoch(model, val_loader)
            self.log_metric(val_metric, "validation_metrics", run)
            # Update the scheduler
            scheduler.step(metrics=val_metric["map"])
            # Evaluate patience
            if val_metric["map"] > best_map:
                best_map = val_metric["map"]
                best_epoch = epoch
                patience = 0
                del best_checkpoint
                best_checkpoint = deepcopy(model.state_dict())
            else:
                patience += 1
                if patience > hyperparams["patience"]:
                    break
            print(f"Epoch {epoch}: Validation mAP: {val_metric['map']} (best mAP: {best_map} at epoch {best_epoch})")
        run["best_val_map"] = best_map
        run.stop()
        path = os.path.join(model_dir, f"best-checkpoint-{self.id}.bin")
        self.save(best_checkpoint, path)
        del model, optimizer, scheduler, best_checkpoint
        torch.cuda.empty_cache()
        gc.collect()
        return best_map

    def train_one_epoch(self, model, train_loader, optimizer, accumulation_steps):
        torch.cuda.empty_cache()
        model.train()
        losses_tracker = LossesTracker()
        # Iterate through batches
        for step, (images, targets) in enumerate(tqdm(train_loader)):
            # Move to the device
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            # Forward pass
            losses = model(images, targets)
            losses_tracker.update(losses, len(images))
            losses_tracker.current_loss.backward()
            # Accumulate gradients
            if (step + 1) % accumulation_steps == 0 or step == len(train_loader) - 1:
                # Backward pass
                optimizer.step()
                optimizer.zero_grad()
        del images, targets, losses
        return losses_tracker.average()

    @torch.no_grad()
    def evaluate_one_epoch(self, model, loader):
        model.eval()
        map = MeanAveragePrecision(
            max_detection_thresholds=None,
            )
        map.warn_on_many_detections = False
        # Iterate through batches:
        for images, targets in loader:
            # Move to the device
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            # Forward pass
            outputs = model(images)
            # Compute metrics
            map.update(outputs, targets)
            del images, targets, outputs
            torch.cuda.empty_cache()
        metric = map.compute()
        return metric

    def save(self, best_checkpoint, path):
        # save the model
        torch.save({"model_state_dict": best_checkpoint}, path)

    def initialize_model(self, freeze_weights) -> nn.Module:
        #model
        model_initial_weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model", "best-checkpoint-114epoch.bin")
        if not os.path.exists(model_initial_weights_path):
            gdown.download(
                id="1AcrYCBR5-kg91C61boj221t1X_SVX8Hv",  
                output=model_initial_weights_path,
            )
        if freeze_weights:
            model = maskRCNNModelFreeze()
        else:
            model = maskRCNNModel() # TODO: Freeze the weights if we want to train under these conditions
        model.load_state_dict(torch.load(model_initial_weights_path, map_location=self.device, weights_only=False)['model_state_dict'])
        model.to(self.device)
        return model
    
    def log_metric(self, losses, metric_type,run):
        for key, value in losses.items():           
            run[f"{metric_type}/{key}"].append(value)



class LossesTracker:

    def __init__(self):
        self._reset()
        self.current_loss = None


    def update(self, losses, batch_size):
        for name, value in losses.items():
            value = value.cpu()
            self.losses[name] += value * batch_size
        self.current_loss = sum(losses.values()) 
        self.cumulative_batch_size += batch_size  
        
    def average(self):
        average_losses  = {name: loss / self.cumulative_batch_size for name, loss in self.losses.items()}
        self._reset()
        return average_losses

    def _reset(self):
        self.losses = {
            "loss_classifier": tensor(0.),
            "loss_box_reg": tensor(0.),
            "loss_mask": tensor(0.),
            "loss_objectness": tensor(0.),
            "loss_rpn_box_reg": tensor(0.),
        }
        self.current_loss = tensor(0.)
        self.cumulative_batch_size = tensor(0.)