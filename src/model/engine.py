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

from src.model.model import maskRCNNModel

class FitterMaskRCNN():

    def __init__(self, id: str):
        # Set the device between GPU, MPS, or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id = id

    def fit(self, train_loader, val_loader, hyperparams, model_dir, neptune_workspace, neptune_project):
        # Initialize the model
        model = self.initialize_model()
        # Initialize optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        # Set up the neptune experiment
        if f"{neptune_workspace}/{neptune_project}" not in get_project_list():
            print("hi")
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
            losses = self.train_one_epoch(model, train_loader, optimizer)
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
        return best_map

    def train_one_epoch(self, model, train_loader, optimizer):
        model.train()
        losses_tracker = LossesTracker()
        # Iterate through batches
        for images, targets in tqdm(train_loader):
            # Move to the device
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            # Forward pass
            losses = model(images, targets)
            losses_tracker.update(losses, len(images))
            # Backward pass
            optimizer.zero_grad()
            losses_tracker.current_loss.backward()
            optimizer.step()
        return losses_tracker.average()

    def evaluate_one_epoch(self, model, loader):
        model.eval()
        map = MeanAveragePrecision(
            max_detection_thresholds=None,
            )
        map.warn_on_many_detections = False
        # Iterate through batches:
        with torch.no_grad():
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

    def initialize_model(self) -> nn.Module:
        #model
        model_initial_weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model", "best-checkpoint-114epoch.bin")
        if not os.path.exists(model_initial_weights_path):
            gdown.download(
                id="1AcrYCBR5-kg91C61boj221t1X_SVX8Hv",  
                output=model_initial_weights_path,
            )
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
    

### original code form GOAT repo ###


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FitterMaskRCNN_old:
    def __init__(self, model, device, config):
        """
        Engine for Fitting MaskRCNN model. For configs see config.
        :param model: MaskRCNN model
        :param device: torch.device, specified in config
        :param config: config file
        """
        self.config = config
        self.epoch = 0

        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5

        self.device = device
        self.model = model.to(device)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        patience = 0
        losses_train = []
        losses_val = []
        lrs = []
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                lrs.append(lr)
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}\n')

            t = time.time()
            summary_losses = self.train_one_epoch(train_loader)
            losses_train.append(summary_losses[0].avg)
            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_losses[0].avg:.5f} ' + \
                f'loss_classifier: {summary_losses[1].avg:.5f}, ' + \
                f'loss_box_reg: {summary_losses[2].avg:.5f}, ' + \
                f'loss_mask: {summary_losses[3].avg:.5f}, ' + \
                f'loss_objectness: {summary_losses[4].avg:.5f}, ' + \
                f'loss_rpn_box_reg: {summary_losses[5].avg:.5f}, ' + \
                f'time: {(time.time() - t):.5f}'
            )
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_losses = self.validation(validation_loader)
            losses_val.append(summary_losses[0].avg)
            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_losses[0].avg:.5f} ' + \
                f'loss_classifier: {summary_losses[1].avg:.5f}, ' + \
                f'loss_box_reg: {summary_losses[2].avg:.5f}, ' + \
                f'loss_mask: {summary_losses[3].avg:.5f}, ' + \
                f'loss_objectness: {summary_losses[4].avg:.5f}, ' + \
                f'loss_rpn_box_reg: {summary_losses[5].avg:.5f}, ' + \
                f'time: {(time.time() - t):.5f}'
            )

            if summary_losses[0].avg < self.best_summary_loss:
                patience = 0
                print("saving best model")
                self.best_summary_loss = summary_losses[0].avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)
            else:
                patience += 1
                print("patience:", patience)
                if patience > 15:
                    print("//////////////// Patience. Training done.")
                    break

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_losses[0].avg)

            # plot and save train log
            fig, ax = plt.subplots(ncols=1)
            ax.plot(np.arange(len(losses_train)), np.array(losses_train), label="train loss")
            ax.plot(np.arange(len(losses_val)), np.array(losses_val), label="val loss")
            plt.legend()
            plt.grid()
            plt.ylim([0, 2])
            plt.savefig(f'{self.base_dir}/hist.png', dpi=144)
            plt.close(fig)
            np.save(f'{self.base_dir}/train_log.png', np.array([losses_train, losses_val, lrs]))
            self.epoch += 1

    def validation(self, val_loader):
        self.model.train()
        losses = [AverageMeter() for x in range(6)]
        t = time.time()
        for step, (images, targets, path) in enumerate(val_loader):

            if self.config.verbose:
                print(
                    f'Val Step {step}/{len(val_loader)}, ' + \
                    f'summary_loss: {losses[0].avg:.5f}, ' + \
                 #  f'loss_classifier: {losses[1].avg:.5f}, ' + \
                 #  f'loss_box_reg: {losses[2].avg:.5f}, ' + \
                 #  f'loss_mask: {losses[3].avg:.5f}, ' + \
                 #  f'loss_objectness: {losses[4].avg:.5f}, ' + \
                 #  f'loss_rpn_box_reg: {losses[5].avg:.5f}, ' + \
                    f'time: {(time.time() - t):.5f}', end='\r'
                )
            with torch.no_grad():
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                batch_size = len(images)
                output = self.model(images, targets)
                losses[0].update(sum(output.values()).detach().item(), batch_size)
                [losses[ii+1].update(output[k].item(), batch_size) for ii, k in enumerate(output.keys())]

        print("")
        return losses

    def train_one_epoch(self, train_loader):
        accumulation_steps = self.config.accumulation_steps

        self.model.train()
        losses = [AverageMeter() for x in range(6)]
        t = time.time()

        self.optimizer.zero_grad()
        total_loss = 0
        for step, (images, targets, path) in enumerate(train_loader):
            if self.config.verbose:
                if step % accumulation_steps == 0:
                    print(

                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {losses[0].avg:.5f}, ' + \
                     #  f'loss_classifier: {losses[1].avg:.5f}, ' + \
                     #  f'loss_box_reg: {losses[2].avg:.5f}, ' + \
                     #  f'loss_mask: {losses[3].avg:.5f}, ' + \
                     #  f'loss_objectness: {losses[4].avg:.5f}, ' + \
                     #  f'loss_rpn_box_reg: {losses[5].avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            batch_size = len(images)

            output = self.model(images, targets)
            sumloss = sum(output.values())
            (sumloss / accumulation_steps).backward()

            sumloss = sumloss.detach().cpu().numpy()
            total_loss += sumloss

            if (step + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses[0].update(total_loss / accumulation_steps, batch_size)
                [losses[ii+1].update(output[k].item(), batch_size) for ii, k in enumerate(output.keys())]
                total_loss = 0

            if self.config.step_scheduler:
                self.scheduler.step()
        return losses

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        print("Checkpoint loaded for epoch:", self.epoch - 1)

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')