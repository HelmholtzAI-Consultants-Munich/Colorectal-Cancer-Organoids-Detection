import os

import pandas as pd
import numpy as np
import napari
import cv2 

from src.utils import *
from src.const import *

def main():
    # parse teh input arguments
    args = get_args()

    # check the dataset structure
    assert os.path.exists(args.dataset), "The dataset path does not exist."
    images = os.path.join(args.dataset, IMAGES_SUBFOLDER)
    assert os.path.exists(images), "The dataset path does not contain an images subfolder."
    annotations = os.path.join(args.dataset, ANNOTATIONS_SUBFOLDER)
    assert os.path.exists(annotations), "The dataset path does not contain a annotations subfolder."
    annotated_images_file = os.path.join(annotations, ANNOTATE_IMAGES_FILE_ORG)

    # read the paths to the annotated images
    images_paths = []
    bboxes_paths = []
    with open(annotated_images_file, "r") as f:
        for path in f.readlines():
            # remove the \n character
            path = path[:-1]
            path = path.split(',')[0]
            images_paths.append(os.path.join(images, path))
            bboxes_paths.append(os.path.join(annotations, os.path.splitext(path)[0] + BBOXES_SUFF + ".csv"))
            
    # open the images one after the other
    for i, (image_path, bboxes_path) in enumerate(zip(images_paths, bboxes_paths)):
        print(f"Reviewing the image ({i+1}/{len(images_paths)}): {os.path.relpath(image_path, images)}")

        viewer = napari.Viewer(
            title=os.path.relpath(image_path, images) + f" ({i+1}/{len(images_paths)})"
        )

        # load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        image_layer = viewer.add_image(image)

        # load the prediction
        bboxes = pd.read_csv(bboxes_path, sep=',', index_col=0)
        napari_bboxes = bbox_xyxy_to_napari(bboxes=bboxes)
        shapes_layer = viewer.add_shapes(
            data=napari_bboxes,
            face_color='transparent',
            edge_color="magenta",
            edge_width=4,
            name="organoid detection"
        )
        
        # save results
        @viewer.bind_key('s')# when finished
        def save_results(viewer: napari.Viewer):
            dialog = NextDialog()
            if not dialog.exec():
                return
            napari_bboxes = viewer.layers["organoid detection"].data
            
            relative_path = os.path.relpath(image_path, images)
            os.makedirs(os.path.dirname(bboxes_path), exist_ok=True)
            # write the bounding boxes
            bboxes = bbox_napari_to_xyxy(napari_bboxes=napari_bboxes)
            bboxes.to_csv(bboxes_path)
            viewer.close()
        
        @viewer.bind_key('e')
        def stop_annotating(viewer: napari.Viewer):
            dialog = CloseDialog()
            if dialog.exec():
                exit(1)

        # run napari
        napari.run()

    # save the changes

if __name__=="__main__":
    main()