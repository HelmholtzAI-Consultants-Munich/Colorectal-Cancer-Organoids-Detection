import os
import warnings
import time

import napari
import skimage
import cv2
import numpy as np
import pandas as pd
import torch

from .utils import *
from .const import *



def main():
    # parse the arguments
    args = get_args()

    # check the dataset structure
    assert os.path.exists(args.dataset), "The dataset path does not exist."
    images = os.path.join(args.dataset, IMAGES_SUBFOLDER)
    assert os.path.exists(images), "The dataset path does not contain an images subfolder."


    # extract the paths of the images and the ored8icitons in the dataset
    ext = ['tif', 'png', 'jpg']
    images_paths = []
    predictions_paths = []
    for root, _, files in os.walk(top=images):
        for file in files:
            if file.split('.')[-1] in ext and file[0]!='.':
                file_path = os.path.join(root, file)
                images_paths.append(file_path)


      
    # create the annotations folder and the log file 
    annotations = os.path.join(args.dataset, ANNOTATIONS_SUBFOLDER)
    os.makedirs(annotations, exist_ok=True)
    annotated_images = os.path.join(annotations, ANNOTATE_IMAGES_FILE_FIB)

    # start annotating the images
    for i, image_path in enumerate(images_paths):
    # for image_path, prediction_path in zip(images_paths, predictions_paths):
        if check_element_in_file(annotated_images, os.path.relpath(image_path, images)):
            continue

        start = time.time()
        print(f"Annotating the image ({i+1}/{len(images_paths)}): {os.path.relpath(image_path, images)}")

        viewer = napari.Viewer(
            title=os.path.relpath(image_path, images) + f" ({i+1}/{len(images_paths)})"
        )

        # load the image
        # image = skimage.io.imread(image_path, as_gray=True)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_layer = viewer.add_image(image)

        points_layer = viewer.add_points(
            data=[],
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
            points_path = os.path.join(annotations, os.path.splitext(relative_path)[0] + POINTS_SUFF + ".csv")
            os.makedirs(os.path.dirname(points_path), exist_ok=True)
            # write the points
            points = pd.DataFrame(data=napari_points, columns=["x", "y"]).astype(np.int16)
            points.to_csv(points_path)
            # add the files to the already processed ones
            with open(annotated_images, "a") as f:
                f.write(os.path.relpath(image_path, images) + "," + str(time.time() - start) + '\n') # independet of the cwd
            viewer.close()
        
        @viewer.bind_key('e')
        def stop_annotating(viewer: napari.Viewer):
            dialog = CloseDialog()
            if dialog.exec():
                exit(1)


        # run napari
        napari.run()



if __name__ == "__main__":
    main()