import yaml
import os
from matplotlib import pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2

from src.model.dataset import MaskRCNNDataset
from src.model.model import maskRCNNModel
from src.model.engine import FitterMaskRCNN

from src.scripts.train import get_args

def draw_boxes(image, target, prediction, output_dir, image_id):
    # draw the predictions
    image = (image * 255).detach().cpu().numpy().astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    image_1 = image.copy()
    image_2 = image.copy()
    image = np.ascontiguousarray(image)
    image_1 = np.ascontiguousarray(image_1)
    image_2 = np.ascontiguousarray(image_2)
    image = cv2.putText(image, "Original", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
    # draw the ground truth
    for box in target["boxes"]:
        box = box.cpu().numpy().astype(np.int32)
        image_1 = cv2.rectangle(image_1, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    image_1 = cv2.putText(image_1, "Ground Truth", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
    # draw the predictions
    for box in prediction["boxes"]:
        box = box.cpu().numpy().astype(np.int32)
        image_2 = cv2.rectangle(image_2, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    image_2 = cv2.putText(image_2, "Predictions", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
    # save the images
    cv2.imwrite(os.path.join(output_dir, f"{image_id}.jpg"), np.concatenate([image, image_1, image_2], axis=1))


def draw_masks(image, target, prediction, output_dir, image_id):
    # draw the predictions
    image = (image * 255).detach().cpu().numpy().astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    image_1 = image.copy()
    image_2 = image.copy()
    image = np.ascontiguousarray(image)
    image_1 = np.ascontiguousarray(image_1)
    image_2 = np.ascontiguousarray(image_2)
    image = cv2.putText(image, "Original", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
    # draw the ground truth
    for box, mask in zip(target["boxes"], target["masks"]):
        box = box.cpu().numpy().astype(np.int32)
        mask = mask.cpu().numpy().astype(np.uint8)
        mask = np.transpose(mask, (1, 2, 0))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        image_1 = cv2.addWeighted(image_1, 1, mask, 0.5, 0)
    image_1 = cv2.putText(image_1, "Ground Truth", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
    # draw the predictions
    for box, mask in zip(prediction["boxes"], prediction["masks"]):
        box = box.cpu().numpy().astype(np.int32)
        mask = mask.cpu().numpy().astype(np.uint8)
        mask = np.transpose(mask, (1, 2, 0))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        image_2 = cv2.addWeighted(image_2, 1, mask, 0.5, 0)
    image_2 = cv2.putText(image_2, "Predictions", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
    # save the images
    cv2.imwrite(os.path.join(output_dir, f"{image_id}.jpg"), np.concatenate([image, image_1, image_2], axis=1))

def main():
    args = get_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = MaskRCNNDataset(config["dataset"], datatype="eval")
    collate_fn = lambda x: tuple(zip(*x))
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    model = maskRCNNModel()
    model.load_state_dict(torch.load(config["model_weights"], map_location=device, weights_only=False)['model_state_dict'])

    engine = FitterMaskRCNN()
    predictions, metric = engine.evaluate_one_epoch_predictions(model, test_loader, config["confidence_threshold"])

    os.makedirs(config["output_dir"], exist_ok=True)

    # draw the predictions
    for i in range(len(predictions)):
        image, target = test_dataset[i]
        draw_boxes(image, target, predictions[i], config["output_dir"], i)
    
    print(metric)




if __name__ == '__main__':
    main()