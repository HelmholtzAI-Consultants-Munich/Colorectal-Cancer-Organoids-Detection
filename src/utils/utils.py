import os
from typing import List

import gdown
import torch

from .const import *

def get_images_paths(dataset_path) -> List[str]:
    """ Get the paths of the images in the dataset.

    :param dataset_path: path to the dataset
    :type dataset_path: str
    :return: list of paths to the images
    :rtype: List[str]
    """
    ext = ['tif', 'tiff']
    images_paths = []
    for root, _, files in os.walk(top=os.path.join(dataset_path, IMAGES_SUBFOLDER)):
        for file in files:
            if file.split('.')[-1] in ext and file[0]!='.':
                file_path = os.path.join(root, file)
                images_paths.append(file_path)
    return images_paths

def get_annotations_paths(images_paths: List[str], dataset_path: str) -> List[str]:
    """ Get the paths of the annotations files from the images paths.

    :param images_paths: list of paths to the images
    :type images_paths: List[str]
    :param dataset_path: path to the dataset
    :type dataset_path: str
    :return: list of paths to the annotations files
    :rtype: List[str]
    """
    annotations_paths = []
    for image_path in images_paths:
        image_rel_path = os.path.relpath(image_path, os.path.join(dataset_path, IMAGES_SUBFOLDER))
        annotations_rel_path = image_to_annotations_path(image_rel_path)
        annotations_path = os.path.join(dataset_path, ANNOTATIONS_SUBFOLDER, annotations_rel_path)
        assert os.path.exists(annotations_path), f"The annotations file {annotations_path} does not exist."
        annotations_paths.append(annotations_path)
    return annotations_paths


def get_images_paths_generic(dataset_path: str):
    """Return path to the images contained in a generic dataset, without a predefined structure.

    :param dataset_path: path to the dataset
    :type dataset_path: str
    :return: list of paths to the images
    :rtype: List[str]
    """
    ext = ['tif', 'png', 'jpg', 'tiff', 'jpeg']
    images_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.split('.')[-1] in ext and file[0]!='.':
                file_path = os.path.join(root, file)
                images_paths.append(file_path)
    return images_paths




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
    images_paths = get_images_paths(dataset_path)
    annotations_paths = get_annotations_paths(images_paths, dataset_path)
    assert len(images_paths) == len(annotations_paths), f"The number of images and annotations files is different."
    # Check for piarwise correspondence between images and annotations
    images_rel_paths = [os.path.relpath(image_path, os.path.join(dataset_path, IMAGES_SUBFOLDER)) for image_path in images_paths]
    annotations_rel_paths = [os.path.relpath(annotations_path, os.path.join(dataset_path, ANNOTATIONS_SUBFOLDER)) for annotations_path in annotations_paths]
    for image_rel_path in images_rel_paths:
        annotations_rel_path = image_to_annotations_path(image_rel_path)
        assert annotations_rel_path in annotations_rel_paths, f"The annotations file {annotations_rel_path} does not exist."
    # Since annotations and images have the same number of elements, we don't have to check the correspondence in the other direction
    return True

def check_inference_dataset(dataset_path):
    # Check if the dataset path exists
    assert os.path.exists(dataset_path), f"The dataset path {dataset_path} does not exist."
    # Check if the images folder exists
    images_paths = get_images_paths_generic(dataset_path)
    assert len(images_paths) > 0, f"The images folder is empty."
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

def load_pretrained_weights(device: str):
    """Load the pretrained weights of GOAT, and downloas them in case they are not locally available.

    :param device: cuda device
    :type device: str
    :return: model state dicts
    :rtype: dict
    """
    model_weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "weights", "pretrained_model.bin")
    if not os.path.exists(model_weights_path):
        gdown.download(
            id="1ipm8sPnGYfoTwrBgT0BE-cmFfdAQTVCK",  
            output=model_weights_path,
        )
    checkpoint = torch.load(model_weights_path, map_location=device)
    return checkpoint['model_state_dict']

def load_finetuned_weights(device):
    """Load the pretrained weights of GOAT, and downloas them in case they are not locally available.

    :param device: cuda device
    :type device: str
    :return: model state dicts
    :rtype: dict
    """
    model_weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "weights", "fine_tuned_model.bin")
    if not os.path.exists(model_weights_path):
        gdown.download(
            id="1gizDHPguUnRoVl_MpleHSZ1BFArncQS-",  
            output=model_weights_path,
        )
    checkpoint = torch.load(model_weights_path, map_location=device)
    return checkpoint['model_state_dict']