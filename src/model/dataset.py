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

    def __init__(self, dataset_path: str, datatype: str = "train", resize: int = 512):
        """
        Dataset class for MaskRCNN model.

        :param dataset_path: path to the dataset
        :type dataset_path: str
        :param datatype: type of the dataset (train, eval)
        :type datatype: str
        :param resize: size of the image to resize to
        :type resize: int
        """

        check_dataset(dataset_path)
        self.images_paths = get_images_paths(dataset_path)
        self.labels_paths = get_annotations_paths(self.images_paths, dataset_path)
        self.resize = resize

        if datatype == "train":
            self.transforms = self._get_train_transforms()
        elif datatype == "eval":
            self.transforms = self._get_eval_transforms()
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
        annotator = None
        if "annotator" in targets_df.columns and not targets_df.empty:
            annotator = random.choice(targets_df["annotator"].unique())
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
    
    def _get_train_transforms_old(self):
        return A.Compose(
        [
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.20, sat_shift_limit=0.10,
                                     val_shift_limit=0.20, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.5),
            ], p=0.7),

            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.30, scale_limit=[-0.5, 1.0], rotate_limit=45, interpolation=1, # weights3/ -0.5 - 1.0
                                   border_mode=0, always_apply=False, p=1),
                A.RandomResizedCrop(size = (512, 512), scale=(0.8, 1.0), ratio=(0.75, 1.33), interpolation=1,
                                    always_apply=False, p=1),
            ], p=0.5),
            A.ToGray(p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Blur(blur_limit=15, p=0.5),
            A.Resize(width=self.resize, height=self.resize, p=1),
            ToTensorV2(p=1.0),

        ],
        p=1.0,
        bbox_params={'format': 'pascal_voc', 'min_area': 2, 'min_visibility': 0.3, 'label_fields': ['labels']}
    )

    def _get_train_transforms(self):
        return A.Compose(
            [
                # leave the image unchenged with 0.2 probability
                A.Sequential([
                    # geometric tranforms
                    A.OneOf([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                    ], p=1.0), 
                    # color transforms
                    A.OneOf([
                        A.HueSaturationValue(hue_shift_limit=0.20, sat_shift_limit=0.10,
                                        val_shift_limit=0.20, p=0.5),
                        A.RandomBrightnessContrast(brightness_limit=0.2,
                                            contrast_limit=0.2, p=0.5),
                    ], p=1.0),
                    # another transformatiio....
                    A.OneOf([
                        A.Blur(blur_limit=15, p=0.5),
                        A.GaussNoise(p=0.5),
                        A.Sharpen(p=0.5),
                    ], p=1.0),
                    A.RandomCrop(1024, 1024),
                ], p=0.8),
                # Fixed transdormations for all images
                A.ToGray(p=1.0),
                ToTensorV2(p=1.0)
            ],
            bbox_params={'format': 'pascal_voc', 'min_area': 2, 'min_visibility': 0.3, 'label_fields': ['labels']},
        )


    def _get_eval_transforms(self):
        return A.Compose(
        [
            A.ToGray(p=1),
            # A.Resize(height=self.resize, width=self.resize, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params={'format': 'pascal_voc', 'min_area': 2, 'min_visibility': 0, 'label_fields': ['labels']}
    )
  

class InferenceMaskRCNNDataset(MaskRCNNDataset):
    def __init__(self, img_paths: str, resize: int = 512):
        """
        Dataset rapper for inference images.
        :param img_paths: list of image paths
        :type img_paths: list
        :param resize: size of the image to resize to
        :type resize: int
        """

        self.img_paths = img_paths
        self.resize = resize
        self.transforms = self._get_transforms()


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        image = self.load_image(image_path)
        image = self.transforms(image=image)['image']
        meta = {"height": image.shape[1], "width": image.shape[2], "path": image_path}
        return image, meta
    
    def _get_transforms(self):
        return A.Compose(
        [
            A.ToGray(p=1),
            # A.Resize(height=self.resize, width=self.resize, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


#### code form GOAT repo ####


class MaskRCNNDataset_old(Dataset):
    def __init__(self, df, img_paths, gt_boxes, gt_rle_strings, datatype="train"):
        super().__init__()
        assert datatype in ["train", "val", "test", "inference", None]

        self.img_paths = img_paths
        self.gt_boxes = gt_boxes
        self.gt_rle_strings = gt_rle_strings
        assert (len(gt_boxes) == len(gt_rle_strings))
        self.mode = datatype


        if datatype == "train":
            self.transforms = get_train_transforms()
        else:
            self.transforms = get_valid_transforms()

        self.rles = []
        self.df = df

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image_path = self.img_paths[index]

        img = self.load_image(image_path)
        boxes = self.load_boxes(index)
        masks = self.load_masks(index, img.shape[0:2])
        labels = torch.ones((boxes.shape[0]), dtype=torch.int64)
        masks = [*masks]

        while True:
            data = {"image": img, "bboxes": boxes, "masks": masks, "category_id": np.arange(len(boxes))}
            augmented = self.transforms(**data)

            aug_img = augmented['image']
            aug_masks = augmented['masks']
            aug_boxes = augmented['bboxes']

            if len(aug_boxes) > 0:
                break

        img = np.array(aug_img)
        boxes = np.array(aug_boxes)

        # unlinke masks and labels, labelfields are dropped when boxes are dropped
        masks = np.array(aug_masks)[augmented["category_id"]]
        labels = np.array(labels)[augmented["category_id"]]

        new_boxes = []
        for mask, box in zip(masks, boxes):
            if np.sum(mask) > 4:
                y, x = np.where(mask)
                new_boxes.append([max(0, min(x)-1),
                                  max(0, min(y)-1),
                                  min(max(x)+1, mask.shape[0]), # todo check here
                                  min(max(y)+1, mask.shape[1])])
            else:
                new_boxes.append(box)
        boxes = np.array(new_boxes)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([index])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes)), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area,
                  "iscrowd": iscrowd}

        img = torch.from_numpy(img).permute(2, 0, 1).float()

        return img.float(), target, {"path": image_path, "kind": self.df[self.df.path == image_path]["kind"].values[0]}

    def load_image(self, img_path):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        return image

    def load_boxes(self, index):
        boxess = self.gt_boxes[index].copy()
        assert len(boxess) > 0
        boxess[:, 2] -= 1
        boxess[:, 3] -= 1
        # convert from coco (x,y,w,h) to pascal_voc (x1,y1,x2,y2)
        boxess[:, 2] = boxess[:, 0] + boxess[:, 2]
        boxess[:, 3] = boxess[:, 1] + boxess[:, 3]
        return boxess

    def load_masks(self, index, shape):
        rles = self.gt_rle_strings[index]
        binary = np.zeros((len(rles), *shape), dtype=np.int8)

        for i, rle in enumerate(rles):
            if type(rle) != float:
                binary[i] = rle_decode(rle, shape)
        return binary
    


    


