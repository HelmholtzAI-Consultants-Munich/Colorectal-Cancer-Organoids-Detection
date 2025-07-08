import os

import cv2
import pandas as pd
import torch
import time
from matplotlib.colors import rgb2hex
import gdown

from src.utils.annotation_utils import *
from src.utils.utils import *
from src.utils.const import *
from src.utils.data_utils import *
from src.model.model import maskRCNNModel
from src.model.dataset import InferenceMaskRCNNDataset
from src.annotation.napari_launcher import napari_launcher
from src.annotation.masks_generator import MasksGenerator

def main():
    # parse the arguments
    args = get_args()

    # check the dataset structure
    check_dataset(args.dataset)
    images_path = os.path.join(args.dataset, IMAGES_SUBFOLDER)
    annotations_path = os.path.join(args.dataset, ANNOTATIONS_SUBFOLDER)


    # extract the paths of the images and the predictions in the dataset
    inference_ds = InferenceMaskRCNNDataset(args.dataset)
    corrected_images = os.path.join(annotations_path ,CORRECTED_IMAGES_FILE_ORG)

    # load the model to predict the masks
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "weights", "best-checkpoint-fine-tune.bin")
    if not os.path.exists(model_weights_path):
        gdown.download(
            id="1ipm8sPnGYfoTwrBgT0BE-cmFfdAQTVCK",  
            output=model_weights_path,
        )
    model = maskRCNNModel()
    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    mask_generator = MasksGenerator(model=model, device=device)


    # start annotating the images
    for i, (image, meta) in enumerate(inference_ds):
    # for image_path, prediction_path in zip(images_paths, predictions_paths):
        image_path = meta["path"]
        image_rel_path = os.path.relpath(image_path, images_path)
        if check_element_in_file(corrected_images, image_rel_path) and args.annotate:
            continue
        elif not check_element_in_file(corrected_images, image_rel_path) and not args.annotate:
            continue
        start = time.time()
        print(f"Annotating the image ({i+1}/{len(inference_ds)}): {image_rel_path}")
        title=image_rel_path + f" ({i+1}/{len(inference_ds)})"
        # load the prediction
        bboxes_xyxy = pd.read_csv(os.path.join(annotations_path, image_to_annotations_path(image_rel_path, BBOXES_SUFF)), sep=',', index_col=0)


        # load the image
        # image = skimage.io.imread(image_path, as_gray=True)
        annotations = napari_launcher(
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE),
            bboxes_xyxy=bboxes_xyxy,
            title=title,
        )

        # generate masks
        masks = mask_generator.run(image=image, bboxes=annotations)
        masks = [run_length_encode(mask) for mask in masks]
        annotations["mask"] = masks
        # save annotations
        annotation_path = os.path.join(annotations_path, image_to_annotations_path(image_rel_path, BBOXES_SUFF))
        os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
        annotations.to_csv(annotation_path)


        if args.annotate:
            # add the files to the already processed ones
            with open(corrected_images, "a") as f:
                f.write(image_rel_path + "," + str(time.time() - start) + '\n') # independet of the cwd



if __name__ == "__main__":
    main()