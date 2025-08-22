import os
import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch

from src.utils.const import *
from src.utils.utils import *
from src.utils.data_utils import *
from src.utils.annotation_utils import clip_xyxy_to_image


class MaskRCNNDataset(Dataset):

    def __init__(self, dataset_path: str, datatype: str = "train", data_augmentation: bool = True, annotator: int = None):
        """
        Dataset class for MaskRCNN model.

        :param dataset_path: path to the dataset
        :type dataset_path: str
        :param datatype: type of the dataset (train, eval)
        :type datatype: str
        :param data_augmentation: whether to apply data augmentation
        :type data_augmentation: bool
        :param annotator: annotator to use for the dataset, if None, all annotators are used
        :type annotator: str or None
        """

        check_dataset(dataset_path)
        self.images_paths = get_images_paths(dataset_path)
        self.labels_paths = get_annotations_paths(self.images_paths, dataset_path)
        self.annotator = annotator

        if datatype == "train" and data_augmentation:
            self.transforms = self._get_augmentation_transforms()
        elif datatype == "eval" or not data_augmentation:
            self.transforms = self._get_base_transforms()
        else:
            raise ValueError(f"Unknown datatype {datatype}")


    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        label_path = self.labels_paths[index]
        image = self.load_image(image_path)
        targets_df = pd.read_csv(label_path, sep=',', index_col=0)

        # sample the annotator if we have multiple ones
        if "annotator" in targets_df.columns and not targets_df.empty:
            if self.annotator is not None:
                targets_df = targets_df[targets_df["annotator"] == self.annotator]
            else:
                # randomly sample an annotator
                # annotator = random.choice(targets_df["annotator"].unique())
                annotator = random.choice([1,2])
                targets_df = targets_df[targets_df["annotator"] == annotator]

        # extract the bounding boxes and the masks tenors
        boxes = self.load_boxes(targets_df, image.shape[:-1])
        masks = self.load_masks(targets_df, image.shape[:-1])
        labels = [1 for _ in range(len(boxes))]

        
        # apply augmentations
        augmented = self.transforms(image=image, bboxes=boxes, masks=masks, labels=labels)
        image = augmented['image']
        boxes = torch.tensor(augmented['bboxes'])
        labels = torch.tensor(augmented['labels'], dtype=torch.int64)
        masks = torch.stack(augmented['masks'], dim=0)
        # masks = masks.unsqueeze(1) # add the channel dimension
        if targets_df.empty:
            masks = torch.empty((0, image.shape[1], image.shape[2]))
            boxes = torch.empty((0, 4))
            labels = torch.ones((boxes.shape[0]), dtype=torch.int64)
        
        # create the target dictionary
        iscrowd = torch.zeros((boxes.shape[0]), dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        targets = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area, "iscrowd": iscrowd}
        return image, targets
    
    def load_image(self, img_path):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)
        image /= 255.0
        return image
    
    def load_boxes(self, targets_df: pd.DataFrame, image_size: Tuple[int]) -> torch.Tensor:
        if targets_df.empty:
            return [] #np.empty((0,4))
        boxes = targets_df[['x1', 'y1', 'x2', 'y2']]
        boxes = clip_xyxy_to_image(boxes, image_size)
        return boxes.values

    def load_masks(self, targets_df: pd.DataFrame, image_size: Tuple[int]) -> torch.Tensor:
        if targets_df.empty:
            return [np.zeros(image_size)]
        rle_masks = targets_df['mask'].values
        masks = []
        for rle in rle_masks:
            masks.append(run_length_decode(rle, image_size))
        return masks
    
    def _get_augmentation_transforms(self):
        return A.Compose(
            [
                # leave the image unchenged with 0.2 probability
                A.Sequential([
                    # geometric tranforms
                    A.OneOf([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                    ], p=0.5), 
                    # color transforms
                    A.OneOf([
                        A.HueSaturationValue(hue_shift_limit=[-2, 2], sat_shift_limit=[-2, 2],
                                        val_shift_limit=[-2, 2], p=0.5),
                        A.RandomBrightnessContrast(brightness_limit=[-0.05, 0.05],
                                            contrast_limit=[-0.05, 0.05], p=0.5),
                    ], p=0.5),
                    # another transformatiio....
                    A.OneOf([
                        A.Blur(blur_limit=[3,5], p=0.5),
                        A.GaussNoise(p=0.5, std_range=[0.01, 0.05]),
                        A.Sharpen(p=0.5, alpha=(0.1, 0.3), lightness=(0.1, 0.3)),
                    ], p=0.5),
                    # A.RandomCrop(1024, 1024),
                ], p=0.3),
                # Fixed transdormations for all images
                A.ToGray(p=1.0),
                ToTensorV2(p=1.0)
            ],
            bbox_params=A.BboxParams(format='pascal_voc', min_area=2, min_visibility=0.3, label_fields=['labels']),
        )


    def _get_base_transforms(self):
        return A.Compose(
        [
            A.ToGray(p=1),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(format='pascal_voc', min_area=2, min_visibility=0, label_fields=['labels'])
    )
  

class InferenceMaskRCNNDataset(MaskRCNNDataset):
    def __init__(self, dataset_path: str):
        """
        Dataset rapper for inference images.
        :param img_paths: list of image paths
        :type resize: int
        """

        check_inference_dataset(dataset_path)
        self.images_paths = get_images_paths_generic(dataset_path)
        self.transforms = self._get_base_transforms()


    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        image = self.load_image(image_path)
        image = self.transforms(image=image)['image']
        metadata = {"height": image.shape[1], "width": image.shape[2], "path": image_path}
        return image, metadata
    
    def _get_base_transforms(self):
        return A.Compose(
            [
                A.ToGray(p=1),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )


    


