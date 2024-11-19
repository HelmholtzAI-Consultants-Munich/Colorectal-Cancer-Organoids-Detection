import os
from typing import List

from .const import *

def get_images_paths(dataset_path) -> List[str]:
    """ Get the paths of the images in the dataset.

    :param dataset_path: path to the dataset
    :type dataset_path: str
    :return: list of paths to the images
    :rtype: List[str]
    """
    ext = ['tif', 'png', 'jpg']
    images_paths = []
    predictions_paths = []
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