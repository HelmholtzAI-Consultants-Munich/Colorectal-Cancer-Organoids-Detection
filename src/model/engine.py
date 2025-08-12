import os
from datetime import datetime
import time
from copy import deepcopy
import subprocess


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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


from src.model.model import maskRCNNModel, maskRCNNModelFreeze
from src.utils.utils import load_pretrained_weights

class FitterMaskRCNN():

    def __init__(self, device: str = None):
        # Set the device between GPU, MPS, or CPU
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, id: str, model, train_dataset, val_dataset, hyperparams, model_dir, neptune_workspace, neptune_project):
        # Initialize the dataloader 
        train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"]//hyperparams["accumulation_steps"], shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=self.collate_fn)
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
        run["local_id"] = id
        print(f"Starting training {id} on device {self.device}")

        # compute the original mAP
        val_metric = self.evaluate_one_epoch(model, val_loader)
        print(f"Original model: Validation mAP: {val_metric['map']}")
        run["original_val_map"] = val_metric["map"]
        
        # Iterate through epochs
        patience = 0
        best_map = 0.
        best_epoch = None
        best_checkpoint = deepcopy(model.state_dict())
        for epoch in range(hyperparams["n_epochs"]):
            # Train one epoch
            print(f"Trial {id}, epoch {epoch}")
            losses = self.train_one_epoch(model, train_loader, optimizer, accumulation_steps=hyperparams["accumulation_steps"])
            self.log_metric(losses, "training_losses", run)
            # Validate
            # print(f"Trial {id}, epoch {epoch}: Validation")
            train_metric = self.evaluate_one_epoch(model, train_loader)
            self.log_metric(train_metric, "training_metrics", run)
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
            # print(f"Epoch {epoch}: Validation mAP: {val_metric['map']} (best mAP: {best_map} at epoch {best_epoch})")
        run["best_val_map"] = best_map
        run.stop()
        path = os.path.join(model_dir, f"best-checkpoint-{id}.bin")
        self.save(best_checkpoint, path)
        del model, optimizer, scheduler, best_checkpoint
        torch.cuda.empty_cache()
        gc.collect()
        return best_map

    def fit_distributed(self, id: str, model, train_loader, val_loader, hyperparams, model_dir, neptune_workspace, neptune_project):
        # Initialize the process group
        model = DDP(model)
        train_loader = DistributedSampler(train_loader, shuffle=True)
        val_loader = DistributedSampler(val_loader, shuffle=False)

        # Call the fit method
        best_map =self.fit(id, model, train_loader, val_loader, hyperparams, model_dir, neptune_workspace, neptune_project)
        return best_map
    

    def train_one_epoch(self, model, train_loader, optimizer, accumulation_steps):
        torch.cuda.empty_cache()
        model.train()
        losses_tracker = LossesTracker()
        # Iterate through batches
        for step, (images, targets) in enumerate(train_loader):
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
        torch.cuda.empty_cache()
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
            # Move outputs to CPU to prevent excessive GPU memory usage
            outputs = [{k: v.detach().cpu() for k, v in o.items()} for o in outputs]
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
            # Compute metrics
            map.update(outputs, targets)
            del images, targets, outputs
        metric = map.compute()
        return metric
    
    @torch.no_grad()
    def evaluate_one_epoch_predictions(self, model, loader, confidence_threshold):
        model.eval()
        map = MeanAveragePrecision(
            max_detection_thresholds=None,
            )
        map.warn_on_many_detections = False
        predictions = []
        # Iterate through batches:
        for images, targets in tqdm(loader):
            # Move to the device
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            # Forward pass
            prediction = model(images)
            prediction = [{k: v.detach().cpu() for k, v in p.items()} for p in prediction]
            map.update(prediction, targets)
            # Filter predictions
            predictions.extend(self.filter_predicitons(prediction, confidence_threshold))
            del images, targets
            torch.cuda.empty_cache()
            metric = map.compute()
        return predictions, metric
    
    @torch.no_grad()
    def inference(self, model, loader, confidence_threshold):
        model.eval()
        predictions = []
        for images, _ in tqdm(loader):
            images = [image.to(self.device) for image in images]
            temp_predictions = model(images)
            temp_predictions = [{k: v.detach().cpu() for k, v in p.items()} for p in temp_predictions]
            predictions.extend(self.filter_predicitons(temp_predictions, confidence_threshold))
            del images
        return predictions
    
    @torch.no_grad()
    def predict_image(self, model, image, confidence_threshold=0.5):
        model.eval()
        # Forward pass
        image = [image.to(self.device)]
        prediction = model(image)
        prediction = [{k: v.detach().cpu() for k, v in p.items()} for p in prediction]
        # Filter predictions
        filtered_predictions = self.filter_predicitons(prediction, confidence_threshold)[0]
        return filtered_predictions
    
    @staticmethod
    def filter_predicitons(predictions, confidence_threshold):
        filtered_predictions = []
        for prediciton in predictions:
            boxes = prediciton["boxes"]
            labels = prediciton["labels"]
            scores = prediciton["scores"]
            masks = prediciton["masks"]
            slices = torch.where(scores > confidence_threshold)
            boxes = boxes[slices]
            labels = labels[slices]
            scores = scores[slices]
            masks = masks[slices]
            # convert the masks to binary masks
            masks = (masks > 0.5).type(torch.uint8)
            filtered_predictions.append({"boxes": boxes, "labels": labels, "scores": scores,"masks": masks})
        return filtered_predictions
    


    def save(self, best_checkpoint, path):
        # save the model
        torch.save({"model_state_dict": best_checkpoint}, path)

    
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