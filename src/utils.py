import argparse

from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QMessageBox
import pandas as pd
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(
        prog="Colorecrtal Cancer Organoids Annotator",
        description="This tool allows to iterate throught a image dataset and correct the already exiting annotatoins.",
    )
    parser.add_argument('-d', '--dataset', help='Path to the folder containing the dataset.')
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

def bbox_napari_to_xyxy(napari_bboxes: list):
    """Convert a bonfing box in napari format to a xyxy format
    """
    bboxes = pd.DataFrame(columns=["x1", "y1", "x2", "y2"], index=list(range(len(napari_bboxes))), dtype=np.int64)
    for i, bbox in enumerate(napari_bboxes):
        bboxes.loc[i] = [bbox[0,0], bbox[0,1], bbox[2,0], bbox[2,1]]
    return bboxes.astype(np.int16)


def bbox_xyxy_to_napari(bboxes: pd.DataFrame):
    """Convert a bounding box in xyx format to napari format.
    """
    napari_bboxes = []
    for _, bbox in bboxes.iterrows():
        napari_bboxes.append(np.array(
            [[bbox["x1"], bbox["y1"]],
                [bbox["x1"], bbox["y2"]],
                [bbox["x2"], bbox["y2"]],
                [bbox["x2"], bbox["y1"]]]
        ))
    return napari_bboxes

def bbox_goat_to_napari(bboxes: np.array):
    """Convert a bounding box in xyx format to napari format.
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
