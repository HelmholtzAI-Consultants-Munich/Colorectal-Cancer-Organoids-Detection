import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision
import numpy as np
import cv2
import src.model.mrcnn_utils as mrcnn_utils

def maskRCNNModel() -> MaskRCNN:
    """Generates a MaskRCNN model with a resnet34 backbone and the same hyperparameters form GOAT.

    :return: MaskRCNN model
    :rtype: MaskRCNN
    """
    backbone = resnet_fpn_backbone(backbone_name='resnet34', weights='IMAGENET1K_V1', trainable_layers=4)
    model = MaskRCNN(backbone,
                      num_classes=2,
                      box_detections_per_img=1000,
                      box_nms_thresh=0.4,
                     )
    model.roi_heads.batch_size_per_image = 256
    model.rpn.batch_size_per_image = 128
    for param in model.parameters():
        param.requires_grad = True
    return model

def maskRCNNModelFreeze() -> MaskRCNN:
    """Generates a MaskRCNN model with a resnet34 backbone and the same hyperparameters form GOAT. 
    In addition, it freezes the backbone, rpn and mask_head.

    :return: MaskRCNN model
    :rtype: MaskRCNN
    """
    model = maskRCNNModel()
    # freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    # freeze mask head
    for param in model.roi_heads.mask_head.parameters():
        param.requires_grad = False
    return model

def fasterRCNNModel(backbone: int, trainable_layers: int) -> torchvision.models.detection.FasterRCNN:
    """Generates a FasterRCNN model with a resnet34 backbone and the same hyperparameters form GOAT.

    :return: FasterRCNN model
    :rtype: FasterRCNN
    """
    backbone = resnet_fpn_backbone(backbone_name=f'resnet{backbone}', weights='IMAGENET1K_V1', trainable_layers=trainable_layers)
    model = torchvision.models.detection.FasterRCNN(backbone,
                                                      num_classes=2,
                                                      box_detections_per_img=1000,
                                                      box_nms_thresh=0.4,
                                                     )
    model.roi_heads.batch_size_per_image = 256
    model.rpn.batch_size_per_image = 128
    for param in model.parameters():
        param.requires_grad = True
    return model

def fasterRCNNModelFreeze(backbone: int, trainable_layers: int) -> torchvision.models.detection.FasterRCNN:
    """Generates a FasterRCNN model with a resnet34 backbone and the same hyperparameters form GOAT.
    In addition, it freezes the backbone.

    :return: FasterRCNN model
    :rtype: FasterRCNN
    """
    model = fasterRCNNModel(backbone=backbone, trainable_layers=trainable_layers)
    # freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    return model
