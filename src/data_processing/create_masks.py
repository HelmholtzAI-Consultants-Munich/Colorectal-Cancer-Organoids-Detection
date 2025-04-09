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
from src.model.dataset import InferenceMaskRCNNDataset


def reshape_bboxes(bboxes: pd.DataFrame, original_size: Tuple[int], new_size: Tuple[int]) -> torch.tensor:
    # size of the image are in the format (height, width)
    tensor = torch.tensor(bboxes.values, dtype=torch.float32)
    if tensor.shape[1] == 1:
        tensor = tensor.T
    x_reshape = new_size[1] / original_size[1]
    y_reshape = new_size[0] / original_size[0]
    tensor[:, 0] *= x_reshape
    tensor[:, 2] *= x_reshape
    tensor[:, 1] *= y_reshape
    tensor[:, 3] *= y_reshape
    return tensor

def predict_masks(image: torch.tensor, bboxes: pd.DataFrame, model: torch.nn.Module, image_size: Tuple[int], device) -> np.ndarray:
    
    if len(bboxes) == 0:
        print(f"no annotations for {bboxes}")
        return torch.empty(0)
    # normalize the image
    image_norm = image.unsqueeze(0).to(device)
    image_norm, _ = model.transform(image_norm, None)
    # run the backbone
    features = model.backbone(image_norm.tensors)
    # adapt the boxes size to the new image shape
    bboxes = reshape_bboxes(bboxes, image_size, (image_norm.image_sizes[0][0], image_norm.image_sizes[0][1]))
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
    detections = model.transform.postprocess(detections, image_norm.image_sizes, [image_size])
    return detections[0]["masks"].squeeze(1).detach().cpu()
    
def main():

    #parse the input arguments
    parser = argparse.ArgumentParser(
        prog="Colorectal Cancer Organoids Annotator",
        description="This tool allows to predict the detection masks from the bounding boxes.",
    )
    parser.add_argument('-d', '--dataset', help='Path to the folder containing the annotated dataset.', required=True)
    parser.add_argument('-o', '--output', help='Path to the output folder.', required=True)
    args = parser.parse_args()

    images_dir = os.path.join(args.dataset, IMAGES_SUBFOLDER)
    annotations_dir = os.path.join(args.dataset, ANNOTATIONS_SUBFOLDER)

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    shutil.copytree(os.path.join(args.dataset, IMAGES_SUBFOLDER), os.path.join(args.output, IMAGES_SUBFOLDER))

    # extract the paths of the images and the predictions in the dataset

    inference_ds = InferenceMaskRCNNDataset(dataset_path=args.dataset)

    # load the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = maskRCNNModel()
    model_weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model", "best-checkpoint-114epoch.bin")
    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 1. Predict the masks
    for image, meta in tqdm(inference_ds):
        annotations_rel_path = image_to_annotations_path(os.path.relpath(meta["path"], images_dir), BBOXES_SUFF)
        annotations_path = os.path.join(annotations_dir, annotations_rel_path)
        annotations = pd.read_csv(annotations_path, sep=',', index_col=0)
        bboxes = annotations[['x1', 'y1', 'x2', 'y2']]
        masks = predict_masks(image, bboxes, model, (meta["height"], meta["width"]), device)
        mask_rles = [run_length_encode(mask) for mask in masks]
        annotations["mask"] = mask_rles
        annotation_output_path = os.path.join(args.output, ANNOTATIONS_SUBFOLDER, annotations_rel_path)
        os.makedirs( os.path.dirname(annotation_output_path), exist_ok=True)
        annotations.to_csv(os.path.join(args.output, ANNOTATIONS_SUBFOLDER, annotations_rel_path))
    print("The masks have been generated successfully.")
    
if __name__ == "__main__":
    main()