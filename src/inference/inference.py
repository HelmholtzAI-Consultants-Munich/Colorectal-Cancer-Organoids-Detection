# script to run inference on the images captured during an experiment

import argparse
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import cv2

from src.utils.data_utils import run_length_encode
from src.model.dataset import InferenceMaskRCNNDataset
from src.model.engine import FitterMaskRCNN
from src.model.model import maskRCNNModel
from src.utils.data_utils import masks_to_area, mask_to_eccentricity, mask_to_contour

def process_predictions(predictions):
    """
    Process the predictions to extract the area of the masks.
    """
    df = pd.DataFrame(columns=["score", "area", "eccentricity", "x1", "y1", "x2", "y2", "mask"])
    for i, (box, mask, label, score) in enumerate(zip(predictions["boxes"], predictions["masks"], predictions["labels"], predictions["scores"])):
        # Filter the predictions based on the confidence threshold
        box = box.tolist()
        mask_rle = run_length_encode(mask)
        score = score.item()
        area = masks_to_area(mask)
        eccentricity = mask_to_eccentricity(mask)
        df.loc[i] = [score, area, eccentricity, box[0], box[1], box[2], box[3], mask_rle]
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
    well_name = image_name.split("_")[-1].replace(" ", "")
    
    return experiment_name, well_name

def get_args():
    parser = argparse.ArgumentParser(
        prog="Colorectal Cancer Organoids Annotator",
        description="This tool allows to evaluate the images captured during an experiment.",
    )
    parser.add_argument('-d', '--dataset', help='Path to the folder containing the annotated dataset.', required=True)
    parser.add_argument('-o', '--output', help='Path to the output folder.', required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    run_inference(args.dataset, args.output)
    

def run_inference(dataset: str, output: str):
    # Load the dataset
    dataset = InferenceMaskRCNNDataset(dataset)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn = lambda x: tuple(zip(*x)))
    print(f"Loaded {len(dataset)} images from {dataset}")
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = maskRCNNModel()
    model_weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model", "best-checkpoint-fine-tune.bin") # TODO: make it configurable, and download weigts
    model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=False)['model_state_dict'])
    model.to(device)
    
    # Perform evaluation
    engine = FitterMaskRCNN(0)
    predictions = engine.inference(model, loader, 0.9) #TODO: make threshold configurable
    
    # process predictions
    os.makedirs(output, exist_ok=True)
    global_experiment_name, _ = parse_image_name(dataset.images_paths[0])
    output_predictions = os.path.join(output, f"{global_experiment_name}_predictions")
    os.makedirs(output_predictions, exist_ok=True)
    output_visualizations = os.path.join(output, f"{global_experiment_name}_visualizations")
    os.makedirs(output_visualizations, exist_ok=True)
    data = []
    for image_path, prediciton in zip(dataset.images_paths, predictions):
        area = masks_to_area(prediciton["masks"])
        experiment_name, well_name = parse_image_name(image_path)
        if experiment_name != experiment_name:
            RuntimeWarning(f"Experiment name {experiment_name} does not match the global experiment name {global_experiment_name}.")
            os.makedirs(os.path.join(output, experiment_name), exist_ok=True)
        data.append([experiment_name, well_name, area])
        # Save the prediciton
        processed_predictions = process_predictions(prediciton)
        processed_predictions.to_csv(os.path.join(output_predictions,f"{global_experiment_name}_{well_name}.csv"))
        # visualize the predictions
        image = cv2.imread(image_path)
        for mask in prediciton["masks"]:
            contour = mask_to_contour(mask)
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
        cv2.imwrite(os.path.join(output_visualizations, f"{global_experiment_name}_{well_name}.png"), image)

    
    # Save the results to the output folder
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output, f"{global_experiment_name}_organoids_area.txt"), index=False, header=False, sep='\t')

   

if __name__ == "__main__":
    main()