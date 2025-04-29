# script to run inference on the images captured during an experiment

import argparse
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import cv2
from tqdm import tqdm
import shutil

from src.utils.data_utils import run_length_encode
from src.model.dataset import InferenceMaskRCNNDataset
from src.model.engine import FitterMaskRCNN
from src.model.model import maskRCNNModel
from src.utils.data_utils import masks_to_area, mask_to_eccentricity, mask_to_contour, fill_empty_masks

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

def get_args():
    parser = argparse.ArgumentParser(
        prog="Colorectal Cancer Organoids Annotator",
        description="This tool allows to evaluate the images captured during an experiment.",
    )
    parser.add_argument('-d', '--dataset', help='Path to the folder containing the annotated dataset.', required=True)
    parser.add_argument('-o', '--output', help='Path to the output folder.', required=True)
    parser.add_argument('-c', '--confidence', help='Confidence threshold for the predictions.', default=0.5, type=float)
    parser.add_argument('-b', '--batch', help="Number of images that are processed simultaneously.", default=2, type=int)
    parser.add_argument('-p', '--pretrained', help="Use pretrained model", action='store_true')
    parser.set_defaults(pretrained=False)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    run_inference(args.dataset, args.output, args.confidence, args.batch, args.pretrained)
    

def run_inference(dataset: str, output: str, confidence: float, batch_size: int, use_pretrained: bool):
    # Load the dataset
    dataset = InferenceMaskRCNNDataset(dataset)
    print(batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn = lambda x: tuple(zip(*x)))
    print(f"Loaded {len(dataset)} images from {dataset}")
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = maskRCNNModel()
    model_weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model", "best-checkpoint-fine-tune.bin") # TODO: make it configurable, and download weigts
    if use_pretrained:
        # overwrite the weights path
        print("Using pretrained model.")
        model_weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model", "best-checkpoint-114epoch.bin")
    model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=False)['model_state_dict'])
    model.to(device)
    
    
    # process predictions
    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output, exist_ok=True)
    global_experiment_name, _ = parse_image_name(dataset.images_paths[0])
    output_predictions = os.path.join(output, f"{global_experiment_name}_predictions_confidence_{confidence}")
    os.makedirs(output_predictions, exist_ok=True)
    output_visualizations = os.path.join(output, f"{global_experiment_name}_visualizations_confidence_{confidence}")
    os.makedirs(output_visualizations, exist_ok=True)
    data = []
    for image, metadata in tqdm(loader):
        predictions = get_predictions(image, model, device, confidence)
        for prediction, metadata_img in zip(predictions, metadata):
            print(f"Processing {metadata_img['path']}")
            boxes = prediction["boxes"]
            masks = prediction["masks"]
            masks_filled = fill_empty_masks(masks, boxes)
            # Calculate the area of the masks
            area = masks_to_area(masks_filled)
            experiment_name, well_name = parse_image_name(metadata_img["path"])
            if experiment_name != experiment_name:
                RuntimeWarning(f"Experiment name {experiment_name} does not match the global experiment name {global_experiment_name}.")
                os.makedirs(os.path.join(output, experiment_name), exist_ok=True)
            data.append([experiment_name, well_name, area])
            # Save the prediciton
            processed_predictions = process_predictions(prediction)
            processed_predictions.to_csv(os.path.join(output_predictions,f"{global_experiment_name}_{well_name}.csv"))
            # visualize the prediction
            image = cv2.imread(metadata_img["path"])
            for bbox, mask in zip(boxes, masks_filled):
                contour = mask_to_contour(mask)
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)
            cv2.imwrite(os.path.join(output_visualizations, f"{global_experiment_name}_{well_name}.png"), image)

    
    # Save the results to the output folder
    df = pd.DataFrame(data)
    # sort by well name
    df.sort_values(by=1, axis=0, inplace=True)
    df.to_csv(os.path.join(output, f"{global_experiment_name}_organoids_area_confidence_{confidence}.txt"), index=False, header=False, sep='\t')

   

if __name__ == "__main__":
    main()