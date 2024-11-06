import os
import warnings
import time

import napari
import skimage
import cv2
import numpy as np
import pandas as pd
from matplotlib.colors import rgb2hex


from src.utils.annotation_utils import *
from src.utils.utils import *
from src.utils.const import *



def main():
    # parse the arguments
    args = get_args()

    # check the dataset structure
    assert os.path.exists(args.dataset), "The dataset path does not exist."
    images = os.path.join(args.dataset, IMAGES_SUBFOLDER)
    assert os.path.exists(images), "The dataset path does not contain an images subfolder."


    # extract the paths of the images and the ored8icitons in the dataset
    images_paths = get_images_paths(args.dataset) 

    # create the annotations folder and the log file 
    annotations = os.path.join(args.dataset, ANNOTATIONS_SUBFOLDER)
    os.makedirs(annotations, exist_ok=True)
    annotated_images = os.path.join(annotations, ANNOTATE_IMAGES_FILE_FIB)

    # start annotating the images
    for i, image_path in enumerate(images_paths):
    # for image_path, prediction_path in zip(images_paths, predictions_paths):
        image_rel_path = os.path.relpath(image_path, images)
        if check_element_in_file(annotated_images, image_rel_path) and args.annotate:
            continue
        elif not check_element_in_file(annotated_images, image_rel_path) and not args.annotate:
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
        if args.annotate:
            napari_points = []
        else:
            points_path = os.path.join(annotations, image_to_annotations_path(image_rel_path, POINTS_SUFF))
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
            points_path = os.path.join(annotations, image_to_annotations_path(relative_path, POINTS_SUFF))
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