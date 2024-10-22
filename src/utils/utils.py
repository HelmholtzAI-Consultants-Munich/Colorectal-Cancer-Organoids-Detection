import argparse
from typing import List
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
    parser.add_argument('-d', '--dataset', help='Path to the folder containing the dataset.')
    parser.add_argument('-a', '--annotate', action='store_true', help='Annotate the dataset mode (Set by default).')
    parser.add_argument('-r', '--review', dest="annotate", action='store_false', help='Review the dataset mode.')
    parser.set_defaults(annotate=True)

    return parser.parse_args()

def get_images_paths(dataset_path):
    ext = ['tif', 'png', 'jpg']
    images_paths = []
    predictions_paths = []
    for root, _, files in os.walk(top=os.path.join(dataset_path, IMAGES_SUBFOLDER)):
        for file in files:
            if file.split('.')[-1] in ext and file[0]!='.':
                file_path = os.path.join(root, file)
                images_paths.append(file_path)
    return images_paths

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
    """Convert a bonfing box in napari format to a xyxy format (napari format has x and y swapped)
    """
    bboxes = pd.DataFrame(columns=["x1", "y1", "x2", "y2"], index=list(range(len(napari_bboxes))), dtype=np.int64)
    for i, bbox in enumerate(napari_bboxes):
        bboxes.loc[i] = [bbox[0,1], bbox[0,0], bbox[2,1], bbox[2,0]]
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


def check_dataset(dataset_path):
    """
    Check if the dataset is correctly formatted.

    :param dataset_path: The path to the dataset.
    :type dataset_path: str
    :return: True if the dataset is correctly formatted, otherwise False.
    :rtype: bool
    """
    # Check if the dataset path exists
    assert os.path.exists(dataset_path), f"The dataset path {dataset_path} does not exist."
    # Check if the annotations folder exists
    assert os.path.exists(os.path.join(dataset_path, ANNOTATIONS_SUBFOLDER)), f"The annotations folder does not exist in the dataset path {dataset_path}."
    # Check if the images folder exists
    assert os.path.exists(os.path.join(dataset_path, IMAGES_SUBFOLDER)), f"The images folder does not exist in the dataset path {dataset_path}."
    # Check if the annotations and images forders contain the same files
    # TODO

    

    return True


def image_to_annotations_path(image_path: str, suffix: str = BBOXES_SUFF):
    """
    Get the realtive path to the annotations file from the image relative path. Here relative is intended with respect 
    to the "annotations" and "images" folders respectively.

    :param image_path: The relative path to the image.
    :type image_path: str
    :return: The relative path to the annotations file.
    :rtype: str
    """
    # Get the image name
    image_name = os.path.basename(image_path)
    # Get the image name without the extension
    image_name_no_ext = os.path.splitext(image_name)[0]
    # Get the annotations file name
    annotations_name = image_name_no_ext + suffix + ".csv"
    # Get the path to the annotations file
    annotations_path = os.path.join(os.path.dirname(image_path), annotations_name)
    return annotations_path

def annotations_to_image_path(annotations_path: str, suffix: str = BBOXES_SUFF):
    """
    Get the realtive path to the image file from the annotations relative path. Here relative is intended with respect 
    to the "annotations" and "images" folders respectively.

    :param annotations_path: The relative path to the annotations file.
    :type annotations_path: str
    :return: The relative path to the image file.
    :rtype: str
    """
    # Get the annotations name
    annotations_name = os.path.basename(annotations_path)
    # Get the annotations name without the extension
    annotations_name_no_ext = os.path.splitext(annotations_name)[0]
    # Get the image file name
    image_name = annotations_name_no_ext.replace(suffix, "") + ".tif"
    # Get the path to the image file
    image_path = os.path.join(os.path.dirname(annotations_path), image_name)
    return image_path

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

