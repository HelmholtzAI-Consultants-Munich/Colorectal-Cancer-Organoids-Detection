# script to run inference on the images captured during an experiment

import argparse
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import cv2
from tqdm import tqdm
import shutil
import random
import time
import warnings

from src.utils.data_utils import run_length_encode
from src.model.dataset import InferenceMaskRCNNDataset, MaskRCNNDataset
from src.model.engine import FitterMaskRCNN
from src.model.model import maskRCNNModel
from src.utils.data_utils import masks_to_area, mask_to_eccentricity, mask_to_contour, fill_empty_masks
from src.annotation.napari_launcher import napari_launcher
from src.annotation.masks_generator import MasksGenerator
from src.utils.const import *
from src.utils.annotation_utils import *
from src.utils.utils import *



def get_args():
    parser = argparse.ArgumentParser(
        prog="Colorectal Cancer Organoids Annotator",
        description="This tool allows to evaluate the images captured during an experiment.",
    )
    parser.add_argument('-d', '--dataset', help='Path to the folder containing the annotated dataset.', required=True)
    parser.add_argument('-o', '--output', help='Path to the output folder.', required=True)
    parser.add_argument('-c', '--confidence', help='Confidence threshold for the predictions.', default=0.5, type=float)
    parser.add_argument('-n', '--number', help='Number of images to correct', type=int, default=10)
    parser.add_argument('-b', '--batch', help="Number of images that are processed simultaneously.", default=2, type=int)
    parser.add_argument('-p', '--pretrained', help="Use pretrained model", action='store_true')
    parser.set_defaults(pretrained=False)
    args = parser.parse_args()
    return args


def process_predictions(predictions):
    """
    Process the predictions to extract the area of the masks.
    """
    df = pd.DataFrame(columns=["score", "area", "ax1", "ax2", "eccentricity", "x1", "y1", "x2", "y2", "mask"])
    for i, (box, mask, label, score) in enumerate(zip(predictions["boxes"], predictions["masks"], predictions["labels"], predictions["scores"])):
        # Filter the predictions based on the confidence threshold
        box = box.tolist()
        mask_rle = run_length_encode(mask)
        score = score.item()
        area = masks_to_area(mask)
        eccentricity, longax, shortax = mask_to_eccentricity(mask)
        df.loc[i] = [score, area, longax, shortax, eccentricity, box[0], box[1], box[2], box[3], mask_rle]
    return df


def parse_image_name(image_path: str):
    """
    Parse the image name to extract the experiment and well names.
    """
    # Extract the experiment name and well name from the image path
    image_name = os.path.basename(image_path)
    image_name = image_name.split("(")[0]
    experiment_name = image_name.split("_")[:-1]
    experiment_name = "_".join(experiment_name)
    well_name = image_name.split("_")[-1].replace(" ", "").replace("-", "")
    
    return experiment_name, well_name

@torch.no_grad()
def get_predictions(image, model, device, confidence):
    """
    Get the predictions for the image using the model.
    """
    model.eval()
    # Move the image to the device
    image = [i.to(device) for i in image]
    # Get the predictions
    prediction = model(image)
    # Move the predictions to the CPU
    prediction = [{k: v.detach().cpu() for k, v in p.items()} for p in prediction]
    # Filter the predictions
    filtered_predictions = FitterMaskRCNN.filter_predicitons(prediction, confidence)
    return filtered_predictions

def create_dataset(dataset_path: str, images_paths: list):
    """
    Create the dataset folder and copy the images to it.
    """
    if os.path.exists(dataset_path):
        # TODO: check if the dataset is already created correctly
        return
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(os.path.join(dataset_path, IMAGES_SUBFOLDER), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, ANNOTATIONS_SUBFOLDER), exist_ok=True)
    for image_path in images_paths:
        new_image_path = os.path.join(dataset_path, IMAGES_SUBFOLDER, os.path.basename(image_path))
        shutil.copy(image_path, new_image_path)


def compute_error(corrected_dataset_path: str) -> Tuple[float, float]:
    """Compute the mean percentage error and mean absolute error from the previously stored errors"""
    prediction_error_file = os.path.join(corrected_dataset_path, ANNOTATIONS_SUBFOLDER, PREDICTION_ERROR_FILE_ORG)
    df = pd.read_csv(prediction_error_file, header=None)
    df.columns = ["image", "area_sape", "count_smape"]    
    df["area_smape"] = df["area_sape"].astype(float)
    df["count_smape"] = df["count_smape"].astype(float)
    area_smape = df["area_smape"].mean()
    count_smape = df["count_smape"].mean()
    
    return area_smape, count_smape


def correct_predicitons(corrected_dataset_path: str, model: torch.nn.Module, confidence: float, device: str):
    """
    Correct the predictions of the model.
    """
    corrected_dataset = InferenceMaskRCNNDataset(corrected_dataset_path)
    engine = FitterMaskRCNN()
    mask_generator = MasksGenerator(model=model, device=device)
    annotated_images = os.path.join(corrected_dataset_path, ANNOTATIONS_SUBFOLDER, ANNOTATE_IMAGES_FILE_ORG)
    prediction_error_file = os.path.join(corrected_dataset_path, ANNOTATIONS_SUBFOLDER, PREDICTION_ERROR_FILE_ORG)
    for i, (image, metadata) in enumerate(corrected_dataset):
        # Get the predictions
        image_path = metadata["path"]
        image_rel_path = os.path.relpath(image_path, os.path.join(corrected_dataset_path, IMAGES_SUBFOLDER))
        # check if the image is already annotated
        if check_element_in_file(annotated_images, image_rel_path):
            print(f"Image ({i+1}/{len(corrected_dataset)}): {image_rel_path} already annotated. Skipping.")
            continue
        print(f"Correcting the image ({i+1}/{len(corrected_dataset)}): {image_rel_path}")
        predictions = engine.predict_image(
            model=model,
            image=image,
            confidence_threshold=confidence,
        )
        bboxes_xyxy = pd.DataFrame(predictions["boxes"], columns=["x1", "y1", "x2", "y2"])
        title = image_rel_path + f" ({i+1}/{len(corrected_dataset)})" 
        start = time.time()
        annotations = napari_launcher(
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE),
            bboxes_xyxy=bboxes_xyxy,
            title=title,
        )
        masks = mask_generator.run(image=image, bboxes=annotations)
        masks_rle = [run_length_encode(mask) for mask in masks]
        annotations["mask"] = masks_rle
        # save annotations
        annotation_path = os.path.join(corrected_dataset_path, ANNOTATIONS_SUBFOLDER, image_to_annotations_path(image_rel_path, BBOXES_SUFF))
        os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
        annotations.to_csv(annotation_path)
        # save the computed areas
        predicted_area = masks_to_area(predictions["masks"])
        corrected_area = masks_to_area(masks)
        predicted_count = len(predictions["boxes"])
        corrected_count = len(masks)
        area_sape = 2 * abs(predicted_area - corrected_area) / (predicted_area + corrected_area) if (predicted_area + corrected_area) > 0 else 0
        count_sape = 2 * abs(predicted_count - corrected_count) / (predicted_count + corrected_count) if (predicted_count + corrected_count) > 0 else 0

        with open(prediction_error_file, "a") as f:
            f.write(image_rel_path + "," + str(area_sape) + "," + str(count_sape) + '\n')
        # add the files to the already processed ones
        with open(annotated_images, "a") as f:
            f.write(image_rel_path + "," + str(time.time() - start) + '\n') # independet of the cwd


def main():
    args = get_args()
    run_inference(args.dataset, args.output, args.confidence, args.number, args.batch, args.pretrained)
    

def run_inference(dataset_path: str, output: str, confidence: float, number: int, batch_size: int, use_pretrained: bool):

    # Load the dataset
    dataset = InferenceMaskRCNNDataset(dataset_path)
    # check the value of n
    if number < 2: #TODO: change to 10
        print(f"Number of images to correct is less than 10. Using 10.")
        Warning(f"Number of images to correct is less than 10. Using 10.")
        number = 2
    if number > len(dataset.images_paths):
        Warning(f"Number of images to correct is greater than the number of images in the dataset. Using {len(dataset.images_paths)}.")
        number = len(dataset.images_paths)

    # sample images to correct
    images_to_correct = random.sample(dataset.images_paths, number)
    print(len(images_to_correct))
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn = lambda x: tuple(zip(*x)))
    print(f"Loaded {len(dataset)} images from {dataset}")
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = maskRCNNModel()
    if use_pretrained:
        # overwrite the weights path
        print("Using pretrained model.")
        model_weights = load_pretrained_weights(device)
    else:
        model_weights = load_finetuned_weights(device)
    model.load_state_dict(model_weights)
    model.to(device)

    # create corrected dataset and correct images
    corrected_dataset_path = os.path.join(output, "corrected_dataset")
    create_dataset(corrected_dataset_path, images_to_correct)
    correct_predicitons(corrected_dataset_path, model, confidence, device)
    correct_images_dataset = MaskRCNNDataset(corrected_dataset_path, datatype="eval", data_augmentation=False)

    # compute errors on the corrected dataset
    area_smape, count_smape = compute_error(corrected_dataset_path)
    
    # process predictions
    os.makedirs(output, exist_ok=True)
    global_experiment_name, _ = parse_image_name(dataset.images_paths[0])
    output_predictions = os.path.join(output, f"{global_experiment_name}_predictions_confidence_{confidence}")
    os.makedirs(output_predictions, exist_ok=True)
    output_visualizations = os.path.join(output, f"{global_experiment_name}_visualizations_confidence_{confidence}")
    os.makedirs(output_visualizations, exist_ok=True)
    corrected_dataset = os.path.join(output, f"{global_experiment_name}_corrected_dataset")
    os.makedirs(corrected_dataset, exist_ok=True)



    data_area = []
    data_count = []
    for image, metadata in tqdm(loader):
        image_path = metadata[0]["path"]
        # Get the predictions
        if image_path in correct_images_dataset.images_paths:
            # load the prediction from the dataset
            i = correct_images_dataset.images_paths.index(image_path)
            _, prediction = correct_images_dataset[i]
        else:
            predictions = get_predictions(image, model, device, confidence)
        for prediction, metadata_img in zip(predictions, metadata):
            print(f"Processing {image_path}")
            boxes = prediction["boxes"]
            masks = prediction["masks"]
            masks_filled = fill_empty_masks(masks, boxes)
            # Calculate the area of the masks
            area = masks_to_area(masks_filled)
            count = len(prediction["boxes"])
            experiment_name, well_name = parse_image_name(metadata_img["path"])
            if experiment_name != experiment_name:
                RuntimeWarning(f"Experiment name {experiment_name} does not match the global experiment name {global_experiment_name}.")
                os.makedirs(os.path.join(output, experiment_name), exist_ok=True)
            data_area.append([experiment_name, well_name, area, area*area_smape]) #TODO: should the mpe be multiplicated by the prediciton value?
            data_count.append([experiment_name, well_name, count, count*count_smape])
            # Save the prediciton
            processed_predictions = process_predictions(prediction)
            processed_predictions.to_csv(os.path.join(output_predictions,f"{global_experiment_name}_{well_name}.csv"))
            # visualize the prediction
            image = cv2.imread(metadata_img["path"])
            for bbox, mask in zip(boxes, masks_filled):
                contour = mask_to_contour(mask)
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
                # cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)
            cv2.imwrite(os.path.join(output_visualizations, f"{global_experiment_name}_{well_name}.png"), image)

    
    # Save the results to the output folder
    df_area = pd.DataFrame(data_area)
    df_count = pd.DataFrame(data_count)
    # sort by well name
    df_area.sort_values(by=1, axis=0, inplace=True)
    df_count.sort_values(by=1, axis=0, inplace=True)
    df_area.to_csv(os.path.join(output, f"{global_experiment_name}_organoids_area_confidence_{confidence}.txt"), index=False, header=False, sep='\t')
    df_count.to_csv(os.path.join(output, f"{global_experiment_name}_organoids_count_confidence_{confidence}.txt"), index=False, header=False, sep='\t')
   

if __name__ == "__main__":
    main()