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
    annotated_images_file = os.path.join(annotations, ANNOTATE_IMAGES_FILE_FIB)

    # read the paths to the annotated images
    images_paths = []
    points_paths = []
    with open(annotated_images_file, "r") as f:
        for path in f.readlines():
            # remove the \n character
            path = path[:-1]
            path = path.split(',')[0]
            images_paths.append(os.path.join(images, path))
            points_paths.append(os.path.join(annotations, os.path.splitext(path)[0] + POINTS_SUFF + ".csv"))
            
    # open the images one after the other
    for i, (image_path, points_path) in enumerate(zip(images_paths, points_paths)):
        print(f"Reviewing the image ({i+1}/{len(images_paths)}): {os.path.relpath(image_path, images)}")

        viewer = napari.Viewer(
            title=os.path.relpath(image_path, images) + f" ({i+1}/{len(images_paths)})"
        )

        # load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        image_layer = viewer.add_image(image)

        
        points = pd.read_csv(points_path, sep=',', index_col=0)
        napari_points = []
        for _, point in points.iterrows():
            napari_points.append(np.array(point))
        points_layer = viewer.add_points(
            data=napari_points,
            face_color='lime',
            name="fibroblast",
            size=20
        )
        
        # save results
        @viewer.bind_key('s')# when finished
        def save_results(viewer: napari.Viewer):
            dialog = NextDialog()
            if not dialog.exec():
                return
            napari_points = viewer.layers["fibroblast"].data
            
            relative_path = os.path.relpath(image_path, images)
            os.makedirs(os.path.dirname(points_path), exist_ok=True)
            # write the points
            points = pd.DataFrame(data=napari_points, columns=["x", "y"]).astype(np.int16)
            points.to_csv(points_path)
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