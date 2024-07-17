import os
import warnings

import napari
import skimage
import cv2
import numpy as np
import pandas as pd
import torch
import time

from .utils import *
from .const import *
from .goat.engine import FitterMaskRCNN
from .goat.model import maskRCNNModel, predict_image
from .goat.dataset import InferenceMaskRCNNDataset


def main():
    # parse the arguments
    args = get_args()

    # check the dataset structure
    assert os.path.exists(args.dataset), "The dataset path does not exist."
    images = os.path.join(args.dataset, IMAGES_SUBFOLDER)
    assert os.path.exists(images), "The dataset path does not contain an images subfolder."
    # predictions = os.path.join(args.dataset, PREDICTIONS_SUBFOLDER)
    # assert os.path.exists(predictions), "The dataset path does not contain a predictions subfolder."
    model_weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model/best-checkpoint-114epoch.bin")
    print(model_weights_path)


    # extract the paths of the images and the ored8icitons in the dataset
    ext = ['tif', 'png', 'jpg']
    images_paths = []
    predictions_paths = []
    for root, _, files in os.walk(top=images):
        for file in files:
            if file.split('.')[-1] in ext and file[0]!='.':
                file_path = os.path.join(root, file)
                images_paths.append(file_path)
                # look for the relative annotation
                # relative_path = os.path.relpath(file_path, images)
                # prediction_path = os.path.join(predictions, os.path.splitext(relative_path)[0] + ".box")
                # if not os.path.exists(prediction_path):
                #     # create an empty file
                #     with open(prediction_path, "w") as f:
                #         pass
                #     warnings.warn(f"The image {file_path} doesn't have a corresponding prediction.")
                # predictions_paths.append(prediction_path)    
    inference_ds = InferenceMaskRCNNDataset(images_paths)

    

    # load the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = maskRCNNModel()
    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
                
    # create the annotations folder and the log file 
    annotations = os.path.join(args.dataset, ANNOTATIONS_SUBFOLDER)
    os.makedirs(annotations, exist_ok=True)
    annotated_images = os.path.join(annotations, ANNOTATE_IMAGES_FILE_ORG)


    # start annotating the images
    for i, (image_norm, meta) in enumerate(inference_ds):
    # for image_path, prediction_path in zip(images_paths, predictions_paths):
        image_path = meta["path"]
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

        # load the prediction
        predictions, _, _ = predict_image(
            model=model,
            img=image_norm,
            threshold=CONFIDENCE_THRESHOLD,
            orig_shape=(meta["height"], meta["width"]),
        )
        bboxes = bbox_goat_to_napari(bboxes=predictions)
        shapes_layer = viewer.add_shapes(
            data=bboxes,
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
            bboxs_path = os.path.join(annotations, os.path.splitext(relative_path)[0] + BBOXES_SUFF + ".csv")
            os.makedirs(os.path.dirname(bboxs_path), exist_ok=True)
            # write the bounding boxes
            bboxes = bbox_napari_to_xyxy(napari_bboxes=napari_bboxes)
            bboxes.to_csv(bboxs_path)
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