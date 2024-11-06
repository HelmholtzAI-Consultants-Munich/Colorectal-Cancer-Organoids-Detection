import argparse
import os
import shutil   
from typing import List, Tuple

import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from src.utils.annotation_utils import *
from src.utils.utils import *
from src.utils.const import *

def annotations_files(dataset_path: str):
    ext = ['csv']
    annotations_paths = []
    for root, _, files in os.walk(top=os.path.join(dataset_path, ANNOTATIONS_SUBFOLDER)):
        for file in files:
            if file.split('.')[-1] in ext and file[0]!='.':
                file_path = os.path.join(root, file)
                annotations_paths.append(file_path)
    return annotations_paths

def read_annotations(file_path):
    """
    Read the annotations from the file.

    :param file_path: The path to the file.
    :return: The list of annotations.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    annotations = pd.read_csv(file_path, sep=',', index_col=0)
    
    return bbox_xyxy_to_box(annotations)

def average_bbox(bbox_1, bbox_2):
    """
    Compute the average bounding box between two bounding boxes.

    :param bbox_1: The first bounding box.
    :type bbox_1: Box
    :param bbox_2: The second bounding box.
    :type bbox_2: Box
    :return: The average bounding box.
    :rtype: Box
    """

    left = (bbox_1.left + bbox_2.left)/2
    top = (bbox_1.top + bbox_2.top)/2
    right = (bbox_1.right + bbox_2.right)/2
    bottom = (bbox_1.bottom + bbox_2.bottom)/2
    bbox_average = Box(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
    )
    return bbox_average


def merge_annotations(bboxes_1: List[Box], bboxes_2: List[Box], iou_threshold: float, keep_unmatched: bool) -> Tuple[List[Box], List[Box]]:
    """
    Merge the annotations of two annotators.

    :param bboxes_1: The annotations of the first annotator.
    :type bboxes_1: List[Box]
    :param bboxes_2: The annotations of the second annotator.
    :type bboxes_2: List[Box]
    :param iou_threshold: The intersection over union threshold.
    :type iou_threshold: float
    :param keep_unmatched: Keep the unmatched annotations.
    :type keep_unmatched: bool
    :return: The merged annotations.
    :rtype: List[Box]
    """
    iou_matrix = np.zeros((len(bboxes_1), len(bboxes_2)))
    for i, box1 in enumerate(bboxes_1):
        for j, box2 in enumerate(bboxes_2):
            iou = compute_iou(box1, box2)
            if iou > iou_threshold:
                iou_matrix[i, j] = iou
    # match annotations of the same object
    row_indicies, col_indicies = linear_sum_assignment(iou_matrix, maximize=True)
    # delete the annotations that are not matched
    row_indicies_filtered = []
    col_indicies_filtered = []
    for i, j in zip(row_indicies, col_indicies):
        if iou_matrix[i, j] > iou_threshold:
            row_indicies_filtered.append(i)
            col_indicies_filtered.append(j)
    # merge the annotations
    bboxes_matched = []
    bboxes_unmatched = []
    for i, j in zip(row_indicies_filtered, col_indicies_filtered):
        bbox_average = average_bbox(bboxes_1[i], bboxes_2[j])
        bboxes_matched.append(bbox_average)
    if keep_unmatched:
        bboxes_1_unmatched = [bboxes_1[i] for i in range(len(bboxes_1)) if i not in row_indicies_filtered]
        bboxes_2_unmatched = [bboxes_2[j] for j in range(len(bboxes_2)) if j not in col_indicies_filtered]
        bboxes_unmatched.extend(bboxes_1_unmatched)
        bboxes_unmatched.extend(bboxes_2_unmatched)
    return bboxes_matched, bboxes_unmatched

def save_annotations(bboxes_matched, bboxes_unmatched, file_path):
    """
    Save the annotations to a file.

    :param bboxes: The list of annotations.
    :type bboxes: List[Box]
    :param file_path: The path to the file.
    :type file_path: str
    """
    annotations_matched = bbox_box_to_xyxy(bboxes_matched)
    annotations_matched["color"] = "#00ff00"
    annotations_unmatched = bbox_box_to_xyxy(bboxes_unmatched)
    annotations_unmatched["color"] = "#ff0000"
    if len(annotations_matched) > 0 and len(annotations_unmatched) > 0:
        annotations = pd.concat([annotations_matched, annotations_unmatched])
    elif len(annotations_matched) > 0:
        annotations = annotations_matched
    elif len(annotations_unmatched) > 0:
        annotations = annotations_unmatched
    else:
        annotations = annotations_matched
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    annotations.to_csv(file_path)
    



def main():

    #parse the input arguments
    parser = argparse.ArgumentParser(
        prog="Colorectal Cancer Organoids Annotator",
        description="This tool allows to merge the annotations of two separate datasets cureated by two different annotators.",
    )
    parser.add_argument('-a1', '--annotator1', help='Path to the folder containing the first annotated dataset.')
    parser.add_argument('-a2', '--annotator2', help='Path to the folder containing the second annotated dataset.')
    parser.add_argument('-o', '--output', help='Path to the output folder.')
    parser.add_argument('-iou', '--iou', help='The intersection over union threshold.', default=0.5, type=float)
    parser.add_argument('--keep', action='store_true')
    parser.add_argument('--drop', dest='keep', action='store_false')
    parser.set_defaults(keep=True)


    args = parser.parse_args()

    # load the annotations
    annotations_files_1 = annotations_files(args.annotator1)
    annotations_files_2 = annotations_files(args.annotator2)

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    shutil.copytree(os.path.join(args.annotator1, IMAGES_SUBFOLDER), os.path.join(args.output, IMAGES_SUBFOLDER))

    for file_1, file_2 in tqdm(zip(annotations_files_1, annotations_files_2)):
        relative_path_1 = os.path.relpath(file_1, os.path.join(args.annotator1, ANNOTATIONS_SUBFOLDER))
        relative_path_2 = os.path.relpath(file_2, os.path.join(args.annotator2, ANNOTATIONS_SUBFOLDER))
        assert relative_path_1 == relative_path_2, f"The files {relative_path_1} and {relative_path_2} do not match."

        # read the annotations
        bboxes_1 = read_annotations(file_1)
        bboxes_2 = read_annotations(file_2)

        # merge the annotations
        bboxes_matched, bboxes_unmatched = merge_annotations(bboxes_1, bboxes_2, args.iou, args.keep)

        # save the annotations
        save_annotations(bboxes_matched, bboxes_unmatched, os.path.join(args.output,ANNOTATIONS_SUBFOLDER, relative_path_1))
        with open(os.path.join(args.output, ANNOTATIONS_SUBFOLDER, ANNOTATE_IMAGES_FILE_ORG), 'a') as f:
            f.write(f"{annotations_to_image_path(relative_path_1, BBOXES_SUFF)},0\n")
    
    print("The annotations have been merged successfully.")




if __name__=="__main__":
    main()