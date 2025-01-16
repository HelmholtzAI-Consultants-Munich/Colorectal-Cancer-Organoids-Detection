import os
import warnings

import napari
import skimage
import cv2
import numpy as np
import pandas as pd
import torch
import time
from matplotlib.colors import rgb2hex
import gdown

from src.utils.annotation_utils import *
from src.utils.utils import *
from src.utils.const import *
from src.model.engine import FitterMaskRCNN
from src.model.model import maskRCNNModel, predict_image
from src.model.dataset import InferenceMaskRCNNDataset


def main():
    # parse the arguments
    args = get_args()

    # check the dataset structure
    assert os.path.exists(args.dataset), "The dataset path does not exist."
    images = os.path.join(args.dataset, IMAGES_SUBFOLDER)
    assert os.path.exists(images), "The dataset path does not contain an images subfolder."
    model_weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model", "best-checkpoint-114epoch.bin")
    if not os.path.exists(model_weights_path):
        
        gdown.download(
            id="1AcrYCBR5-kg91C61boj221t1X_SVX8Hv",  
            output=model_weights_path,
        )


    # extract the paths of the images and the ored8icitons in the dataset
    images_paths = get_images_paths(args.dataset) 
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

        # load the prediction
        if args.annotate:
            predictions, _, _ = predict_image(
                model=model,
                img=image_norm,
                threshold=CONFIDENCE_THRESHOLD,
                orig_shape=(meta["height"], meta["width"]),
            )
            bboxes_napari = bbox_goat_to_napari(bboxes=predictions)
            edge_color = "blue"
        else:
            annotations_path = os.path.join(annotations, image_to_annotations_path(image_rel_path, BBOXES_SUFF))
            bboxes_xyxy = pd.read_csv(annotations_path, sep=',', index_col=0)
            bboxes_napari = bbox_xyxy_to_napari(bboxes=bboxes_xyxy)
            if "color" in bboxes_xyxy.columns:
                edge_color = bboxes_xyxy["color"].to_list()
            else:
                edge_color = "blue"
        shapes_layer = viewer.add_shapes(
                data=bboxes_napari,
                face_color='transparent',
                edge_color=edge_color,
                edge_width=4,
                name="organoid detection"
            )
        shapes_layer.current_edge_color = "magenta"

        # hide the boxes
        @viewer.bind_key('h')# hide the boxes
        def hide_boxes(viewer: napari.Viewer):
            shapes_layer.visible = not shapes_layer.visible
        # save results
        @viewer.bind_key('s')# when finished
        def save_results(viewer: napari.Viewer):
            dialog = NextDialog()
            if not dialog.exec():
                return
            
            napari_bboxes = viewer.layers["organoid detection"].data
            colors = viewer.layers["organoid detection"].edge_color
            colors_hex = [rgb2hex(color) for color in colors]
            
            bboxs_path = os.path.join(annotations, image_to_annotations_path(image_rel_path, BBOXES_SUFF))
            os.makedirs(os.path.dirname(bboxs_path), exist_ok=True)
            # write the bounding boxes
            bboxes = bbox_napari_to_xyxy(napari_bboxes=napari_bboxes)
            bboxes["color"] = colors_hex
            bboxes.to_csv(bboxs_path)
            if args.annotate:
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