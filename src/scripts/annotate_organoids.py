import os
import cv2
import pandas as pd
import torch
import time
import gdown

from src.utils.annotation_utils import *
from src.utils.utils import *
from src.utils.const import *
from src.utils.data_utils import *
from src.model.engine import FitterMaskRCNN
from src.model.model import maskRCNNModel
from src.model.dataset import InferenceMaskRCNNDataset
from src.annotation.napari_launcher import napari_launcher
from src.annotation.masks_generator import MasksGenerator


def main():
    # parse the arguments
    args = get_args()

    # check the dataset structure
    images_path = os.path.join(args.dataset, IMAGES_SUBFOLDER)
    engine = FitterMaskRCNN()


    # extract the paths of the images and the ored8icitons in the dataset
    inference_ds = InferenceMaskRCNNDataset(args.dataset)

    # load the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = maskRCNNModel()
    if args.pretrained:
        model_weights = load_pretrained_weights(device)
    else:
        model_weights = load_finetuned_weights(device)
    model.load_state_dict(model_weights)
    mask_generator = MasksGenerator(model=model, device=device)
                
    # create the annotations folder and the log file 
    annotations_path = os.path.join(args.dataset, ANNOTATIONS_SUBFOLDER)
    os.makedirs(annotations_path, exist_ok=True)
    annotated_images = os.path.join(annotations_path, ANNOTATE_IMAGES_FILE_ORG)



    # start annotating the images
    for i, (image, meta) in enumerate(inference_ds):
    # for image_path, prediction_path in zip(images_paths, predictions_paths):
        image_path = meta["path"]
        image_rel_path = os.path.relpath(image_path, images_path)
        if check_element_in_file(annotated_images, image_rel_path) and args.annotate:
            continue
        elif not check_element_in_file(annotated_images, image_rel_path) and not args.annotate:
            continue
        title = image_rel_path + f" ({i+1}/{len(inference_ds)})"
        start = time.time()
        print(f"Annotating the image ({i+1}/{len(inference_ds)}): {image_rel_path}")


        # load the prediction
        if args.annotate:
            predictions = engine.predict_image(
                model=model,
                image=image,
                confidence_threshold=CONFIDENCE_THRESHOLD,
            )
            bboxes_xyxy = pd.DataFrame(predictions["boxes"], columns=["x1", "y1", "x2", "y2"])
        else:
            annotation_path = os.path.join(annotations_path, image_to_annotations_path(image_rel_path, BBOXES_SUFF))
            bboxes_xyxy = pd.read_csv(annotation_path, sep=',', index_col=0)

        # run napari instance
        annotations = napari_launcher(
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE),
            bboxes_xyxy=bboxes_xyxy,
            title=title,
        )

        # generate masks
        masks = mask_generator.run(image=image, bboxes=annotations)
        masks = [run_length_encode(mask) for mask in masks]
        print(masks)
        annotations["mask"] = masks
        # save annotations
        annotation_path = os.path.join(annotations_path, image_to_annotations_path(image_rel_path, BBOXES_SUFF))
        os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
        annotations.to_csv(annotation_path)
        if args.annotate:
            # add the files to the already processed ones
            with open(annotated_images, "a") as f:
                f.write(image_rel_path + "," + str(time.time() - start) + '\n') # independet of the cwd



if __name__ == "__main__":
    main()