
from typing import List, Tuple
import argparse
import shutil
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torchvision.models.detection.roi_heads import maskrcnn_inference

from src.utils.const import *
from src.utils.utils import *
from src.utils.data_utils import *
from src.model.model import maskRCNNModel
from src.model.dataset import MaskRCNNDataset
from src.annotation.masks_generator import MasksGenerator

def add_empty_mask_column(dataset_path):
    images_paths = get_images_paths(dataset_path=dataset_path)
    annotations_paths = get_annotations_paths(images_paths=images_paths, dataset_path=dataset_path)
    for annotation_path in annotations_paths:
        target = pd.read_csv(annotation_path, sep=',', index_col=0)
        target["mask"] = ["2764700 100" for _ in range(len(target))]
        target.to_csv(annotation_path)
    

def reshape_annotations(annotations: pd.DataFrame, original_size: Tuple[int], new_size: Tuple[int]) -> torch.tensor:
    # size of the image are in the format (height, width)
    tensor = torch.tensor(annotations.values, dtype=torch.float32)
    if tensor.shape[1] == 1:
        tensor = tensor.T
    x_reshape = new_size[1] / original_size[1]
    y_reshape = new_size[0] / original_size[0]
    tensor[:, 0] *= x_reshape
    tensor[:, 2] *= x_reshape
    tensor[:, 1] *= y_reshape
    tensor[:, 3] *= y_reshape
    return tensor

def predict_masks(image: torch.tensor, annotations: pd.DataFrame, model: torch.nn.Module, image_size: Tuple[int], device) -> np.ndarray:
    
    if len(annotations) == 0:
        print(f"no annotations for {annotations}")
        return torch.empty(0)
    # normalize the image
    image_norm = image.unsqueeze(0).to(device)
    image_norm, _ = model.transform(image_norm, None)
    # run the backbone
    features = model.backbone(image_norm.tensors)
    # adapt the boxes size to the new image shape
    bboxes = reshape_annotations(annotations, image_size, (image_norm.image_sizes[0][0], image_norm.image_sizes[0][1]))
    # run the mask head
    mask_features = model.roi_heads.mask_roi_pool(features, [bboxes], image_norm.image_sizes)
    mask_features = model.roi_heads.mask_head(mask_features)
    mask_logits = model.roi_heads.mask_predictor(mask_features)
    labels = [torch.ones(len(bboxes), device=device, dtype=torch.int64)]
    masks_probs = maskrcnn_inference(mask_logits, labels)

    # detections, _ = model.roi_heads(features, [proposals], image_norm.image_sizes)
    detections = [{
        "boxes": bboxes,
        "masks": masks_probs[0],
        "scores": torch.ones(len(bboxes)),
        "labels": labels,
    }]
    
    detections = model.transform.postprocess(detections, image_norm.image_sizes, [image_size[::-1]])
    return detections[0]["masks"].squeeze(1).detach().cpu().numpy()
    
def main():

    #parse the input arguments
    parser = argparse.ArgumentParser(
        prog="Colorectal Cancer Organoids Annotator",
        description="This tool allows to predict the detection masks from the bounding boxes.",
    )
    parser.add_argument('-d', '--dataset', help='Path to the folder containing the annotated dataset.')
    parser.add_argument('-o', '--output', help='Path to the output folder.')
    args = parser.parse_args()

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    shutil.copytree(os.path.join(args.dataset, IMAGES_SUBFOLDER), os.path.join(args.output, IMAGES_SUBFOLDER))

    add_empty_mask_column(args.dataset)
    dataset = MaskRCNNDataset(args.dataset, datatype="eval", data_augmentation=False)

    # load the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = maskRCNNModel()
    model_weights = load_pretrained_weights(device)
    model.load_state_dict(model_weights)
    model.eval()

    # load mask generator
    mask_generator = MasksGenerator(model, device)

    # check for bexes woth swapped coordinates
    for annotation_path in dataset.labels_paths:
        annotation = pd.read_csv(annotation_path, sep=',', index_col=0)
        annotation_2 = annotation.copy()
        annotation_2["x1"] = annotation[['x1', 'x2']].min(axis=1)
        annotation_2["y1"] = annotation[['y1', 'y2']].min(axis=1)
        annotation_2["x2"] = annotation[['x1', 'x2']].max(axis=1)
        annotation_2["y2"] = annotation[['y1', 'y2']].max(axis=1)
        if not annotation.equals(annotation_2):
            print(f"Swapped coordinates in {annotation_path}")
            annotation_2.to_csv(annotation_path)

    # 1. Predict the masks
    for image, target in tqdm(dataset):
        target_df = pd.DataFrame(target["boxes"], columns=["x1", "y1", "x2", "y2"])
        masks = mask_generator.run(image.to(device), target_df)
        mask_rles = [run_length_encode(mask) for mask in masks]
        annotations_path = image_to_annotations_path(dataset.images_paths[target["image_id"].item()])
        annotations_rel_path = os.path.relpath(annotations_path, start=os.path.join(args.dataset, IMAGES_SUBFOLDER))
        annotation_output_path = os.path.join(args.output, ANNOTATIONS_SUBFOLDER, annotations_rel_path)
        os.makedirs( os.path.dirname(annotation_output_path), exist_ok=True)
        target_df["mask"] = mask_rles
        target_df.to_csv(os.path.join(args.output, ANNOTATIONS_SUBFOLDER, annotations_rel_path))
    print("The masks have been generated successfully.")
    
if __name__ == "__main__":
    main()
