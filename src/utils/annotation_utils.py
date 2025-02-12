import argparse
from typing import List, Tuple
import os

from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QMessageBox
import pandas as pd
import numpy as np

from .const import *

def get_args():
    parser = argparse.ArgumentParser(
        prog="Colorecrtal Cancer Organoids Annotator",
        description="This tool allows to iterate throught a image dataset and correct the already exiting annotatoins.",
    )
    parser.add_argument('-d', '--dataset', help='Path to the folder containing the dataset.', required=True)
    parser.add_argument('-a', '--annotate', action='store_true', help='Annotate the dataset mode (Set by default).')
    parser.add_argument('-r', '--review', dest="annotate", action='store_false', help='Review the dataset mode.')
    parser.set_defaults(annotate=True)

    return parser.parse_args()

def check_element_in_file(file_path, element):
    """
    Check if the specified element is in the file.

    :param file_path: The path to the file.
    :param element: The element to search for in the file.
    :return: True if the element is found, otherwise False.
    """
    try:
        with open(file_path, 'r') as file:
            # Read the file content
            content = file.read()
            # Check if the element is in the file content
            if element in content:
                return True
            else:
                return False
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return False
    

class Box:
    """A class to represent a bounding box. The coordinates are in the xyxy format. 
    """

    def __init__(self, left, top, right, bottom):
        """
        Class constructor. left, top, right, bottom are the coordinates of the box, with the origin at the top-left corner.
        Therefore, the top coordinate is smaller than the bottom coordinate.

        :param left: Left coordinate of the box.
        :type left: _type_
        :param top: Top coordinate of the box.
        :type top: _type_
        :param right: Right coordinate of the box.
        :type right: _type_
        :param bottom: Bottom coordinate of the box.
        :type bottom: _type_
        """
        
        assert left < right, f"Left should be smaller than right. Left: {left}, Right: {right}"
        assert top < bottom, f"Top should be smaller than bottom (the origin is in the top-left corner). Top {top}, Bottom: {bottom}"
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def area(self) -> float:
        """
        Calculate the area of the box.

        :return: area of the box.
        :rtype: float
        """
        return (self.right - self.left) * (self.bottom - self.top)
    
    def diameter(self) -> float:
        """
        Calculate the diameter of the box.

        :return: diameter of the box.
        :rtype: float
        """
        return np.sqrt((self.right - self.left)**2 + (self.bottom - self.top)**2)
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Box):
            return False
        return self.left == value.left and self.top == value.top and self.right == value.right and self.bottom == value.bottom
    

def compute_iou(box1: Box, box2: Box) -> float:
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.

    :param box1: First bounding box.
    :type box1: Box
    :param box2: Second bounding box.
    :type box2: Box
    :return: The IoU between the two bounding boxes.
    :rtype: float
    """
    # Calculate intersection area
    intersection_width = min(box1.right, box2.right) - max(box1.left, box2.left)
    intersection_height = min(box1.bottom, box2.bottom) - max(box1.top, box2.top)
    
    if intersection_width <= 0 or intersection_height <= 0:
        return 0
    
    intersection_area = intersection_width * intersection_height
    
    union_area = box1.area() + box2.area() - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou

def bbox_xyxy_to_box(bboxes: pd.DataFrame) -> List[Box]:
    bboxes_box = []
    for _, box in bboxes.iterrows():
        bboxes_box.append(Box(min(box['x1'], box['x2']), min(box['y1'], box['y2']), max(box['x1'], box['x2']), max(box['y1'], box['y2']),))
    return bboxes_box

def bbox_napari_to_xyxy(napari_bboxes: list):
    """Convert a bounding box in napari format to a xyxy format (napari format has x and y swapped)
    """
    bboxes = pd.DataFrame(columns=["x1", "y1", "x2", "y2"], index=list(range(len(napari_bboxes))), dtype=np.int64)
    for i, bbox in enumerate(napari_bboxes):
        bboxes.loc[i] = [
            min(bbox[0,1], bbox[2,1]),
            min(bbox[0,0], bbox[2,0]), 
            max(bbox[0,1], bbox[2,1]),
            max(bbox[0,0], bbox[2,0]), 
        ]
    return bboxes.astype(np.int16)


def bbox_xyxy_to_napari(bboxes: pd.DataFrame):
    """Convert a bounding box in xyxy format to napari format.
    """
    napari_bboxes = []
    for _, bbox in bboxes.iterrows():
        napari_bboxes.append(np.array(
            [[bbox["y1"], bbox["x1"]],
                [bbox["y1"], bbox["x2"]],
                [bbox["y2"], bbox["x2"]],
                [bbox["y2"], bbox["x1"]]]
        ))
    return napari_bboxes

def bbox_goat_to_napari(bboxes: np.array):
    """Convert a bounding box in the format returned by GOAT to napari format.
    """
    napari_bboxes = []
    for bbox in bboxes:
        napari_bboxes.append(np.array(
            [[bbox[1], bbox[0]],
                [bbox[1], bbox[2]],
                [bbox[3], bbox[2]],
                [bbox[3], bbox[0]]]
        ))
    return napari_bboxes

def bbox_goat_to_xyxy(goat_bboxes: np.array):
    """Convert a bounding box in the format returned by GOAT to xyxy format.
    """
    bboxes = pd.DataFrame(columns=["x1", "y1", "x2", "y2"], index=list(range(len(goat_bboxes))), dtype=np.int64)
    for i, bbox in enumerate(goat_bboxes):
        bboxes.loc[i] = [bbox[0], bbox[1], bbox[2], bbox[3]]
    return bboxes.astype(np.int16)

def bbox_box_to_xyxy(bboxes: List[Box]) -> pd.DataFrame:
    """Convert a list of bounding boxes in Box format to a DataFrame in xyxy format.
    """
    bboxes_df = pd.DataFrame(columns=["x1", "y1", "x2", "y2"], index=list(range(len(bboxes))), dtype=np.int64)
    for i, bbox in enumerate(bboxes):
        bboxes_df.loc[i] = [bbox.left, bbox.top, bbox.right, bbox.bottom]
    return bboxes_df.astype(np.int16)

def clip_xyxy_to_image(bboxes: pd.DataFrame, image_shape: Tuple[int, int]) -> pd.DataFrame:
    """Clip the bounding boxes to the image shape. The image shape is in the format (height, width).
    """
    bboxes.loc[:, "x1"] = bboxes["x1"].clip(lower=0, upper=image_shape[1])
    bboxes.loc[:, "x2"] = bboxes["x2"].clip(lower=0, upper=image_shape[1])
    bboxes.loc[:, "y1"] = bboxes["y1"].clip(lower=0, upper=image_shape[0])
    bboxes.loc[:, "y2"] = bboxes["y2"].clip(lower=0, upper=image_shape[0])
    return bboxes

class NextDialog(QDialog):
    """Dialoge window to confirm passing to the next image.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Go to the next image")
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        message = QLabel("Saving and going to the next image in the dataset. Once saved it will not be possible to edit again the current image.\n\nDo you wat to continue?")
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

class CloseDialog(QDialog):
    """Dialoge window to confirm closing the application.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Go to the next image")
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        message = QLabel("Do you wat to close the application?")
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

