from typing import List, Tuple

import pandas as pd
import numpy as np
import torch
from torchvision.models.detection.roi_heads import maskrcnn_inference

from src.utils.utils import *
from src.utils.data_utils import *

class MasksGenerator:

    def __init__(self, model: torch.nn.Module,  device: str = None):
        self.model = model
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

    def run(self, image: torch.tensor, bboxes: pd.DataFrame) -> np.ndarray:
        # create the masks
        bboxes = torch.tensor(bboxes[['x1', 'y1', 'x2', 'y2']].to_numpy(dtype=np.float32), dtype=torch.float32)
        masks = self.predict_masks(image, bboxes)
        masks = fill_empty_masks(masks, bboxes)
        # encode the masks
        # mask_rles = [run_length_encode(mask) for mask in masks]
        return masks

    def reshape_bboxes(self, bboxes: torch.Tensor, original_size: Tuple[int], new_size: Tuple[int]) -> torch.tensor:
        # size of the image are in the format (height, width)
        if bboxes.shape[1] == 1:
            bboxes = bboxes.T
        x_reshape = new_size[1] / original_size[1]
        y_reshape = new_size[0] / original_size[0]
        bboxes[:, 0] *= x_reshape
        bboxes[:, 2] *= x_reshape
        bboxes[:, 1] *= y_reshape
        bboxes[:, 3] *= y_reshape
        return bboxes

    def predict_masks(self, image: torch.Tensor, bboxes: torch.Tensor) -> np.ndarray:
        
        print(f"image shape: {image.shape}")
        image_size = (image.shape[1], image.shape[2])
        if len(bboxes) == 0:
            print(f"no annotations for {bboxes}")
            return torch.empty(0)
        # normalize the image
        image_norm = image.unsqueeze(0).to(self.device)
        image_norm, _ = self.model.transform(image_norm, None)
        # run the backbone
        features = self.model.backbone(image_norm.tensors)
        # adapt the boxes size to the new image shape
        bboxes = self.reshape_bboxes(bboxes, image_size, (image_norm.image_sizes[0][0], image_norm.image_sizes[0][1]))
        # run the mask head
        mask_features = self.model.roi_heads.mask_roi_pool(features, [bboxes], image_norm.image_sizes)
        mask_features = self.model.roi_heads.mask_head(mask_features)
        mask_logits = self.model.roi_heads.mask_predictor(mask_features)
        labels = [torch.ones(len(bboxes), device=self.device, dtype=torch.int64)]
        masks_probs = maskrcnn_inference(mask_logits, labels)[0]
        # detections, _ = model.roi_heads(features, [proposals], image_norm.image_sizes)
        detections = [{
            "boxes": bboxes,
            "masks": masks_probs,
            "scores": torch.ones(len(bboxes)),
            "labels": labels,
        }]
        detections = self.model.transform.postprocess(detections, image_norm.image_sizes, [image_size])
        return detections[0]["masks"].detach().cpu()