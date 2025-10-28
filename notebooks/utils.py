from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch


def compute_iou(box_1, box_2):
    # compute the intersection over union of two bounding boxes
    x1 = max(box_1[0], box_2[0])
    y1 = max(box_1[1], box_2[1])
    x2 = min(box_1[2], box_2[2])
    y2 = min(box_1[3], box_2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box_1 = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    area_box_2 = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])
    
    union = area_box_1 + area_box_2 - intersection
    
    if union == 0:
        return 0.0
    return intersection / union

def compute_iou_matrix(bboxes_1, bboxes_2):
    # compute the iou matrix
    iou_matrix = np.zeros((len(bboxes_1), len(bboxes_2)))
    for i, box_1 in enumerate(bboxes_1):
        for j, box_2 in enumerate(bboxes_2):
            iou_matrix[i, j] = compute_iou(box_1, box_2)
    return iou_matrix


def instance_matcher(iou_matrix, iou_threshold: float = 0.5):
    # filter out the matches with low iou
    iou_matrix[iou_matrix < iou_threshold] = 0
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)
    # fileter out the matches (some of them might be an artifact of the Hungarian algorithm)
    row_ind_threshold = []
    col_ind_threshold = []
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= iou_threshold:
            row_ind_threshold.append(i)
            col_ind_threshold.append(j)
    return row_ind_threshold, col_ind_threshold

def compute_calibration_error(bins):
    # compute the calibration score
    ce = 0.0
    for index, row in bins.iterrows():
        if row["count"] == 0:
            continue
        ce += abs(row["match"] - row["confidence"]) * row["count"]
    ce /= bins["count"].sum()
    return ce



def match_predictions(predictions, dataset, iou_threshold):
    predictions = copy.deepcopy(predictions)
    detections = pd.DataFrame(columns=["image_id", "detection_id", "confidence_score", "match"])
    for image_id, image_predictions in tqdm(enumerate(predictions)):
        _, labels = dataset[image_id]
        iou_matrix = compute_iou_matrix(image_predictions["boxes"].cpu().numpy(), labels["boxes"].cpu().numpy())
        row_ind, col_ind = instance_matcher(iou_matrix, iou_threshold=iou_threshold)
        for detection_id in range(len(image_predictions["boxes"])):
            confidence_score = image_predictions["scores"][detection_id].item()
            match = int(detection_id in col_ind)
            detections.loc[len(detections)] = [image_id, detection_id, confidence_score, match]

    return detections

def match_labels(dataset_1, dataset_2, iou_threshold):
    matched = []
    for image_id in tqdm(range(len(dataset_1))):
        _, label_1 = dataset_1[image_id]
        _, label_2 = dataset_2[image_id]
        iou_matrix = compute_iou_matrix(label_1["boxes"].cpu().numpy(), label_2["boxes"].cpu().numpy())
        row_ind, col_ind = instance_matcher(iou_matrix, iou_threshold=iou_threshold)
        boxes = []
        scores = []
        for r,c in zip(row_ind, col_ind):
            box_1 = label_1["boxes"][r].cpu().numpy()
            box_2 = label_2["boxes"][c].cpu().numpy()
            box = (box_1 + box_2) / 2
            boxes.append(box)
            scores.append(1)
        # add unmatched boxes from label_1
        for r in range(len(label_1["boxes"])):
            if r not in row_ind:
                boxes.append(label_1["boxes"][r].cpu().numpy())
                scores.append(0.5)
        # add unmatched boxes from label_2
        for c in range(len(label_2["boxes"])):
            if c not in col_ind:
                boxes.append(label_2["boxes"][c].cpu().numpy())
                scores.append(0.5)
        if len(boxes) == 0:
            boxes = np.zeros((0,4))
            scores = np.zeros((0))
        matched.append({
            "boxes": torch.tensor(np.array(boxes)),
            "scores": torch.tensor(np.array(scores)),
        })
    return matched

def bin_predictions(detections: pd.DataFrame, bins_extremes: list, labels: list):
    detections = detections.copy()
    detections["bin"] = pd.cut(detections["confidence_score"], bins=bins_extremes, labels=labels)
    bins = detections.groupby("bin", observed=False)["match"].mean().reset_index()
    bins["count"] = detections.groupby("bin", observed=False)["match"].count().values
    bins["confidence"] = detections.groupby("bin", observed=False)["confidence_score"].mean().values
    bins["interval"] = bins_extremes[1:]
    return bins

def plot_calibration(detections: dict, dataset: str, iou: float, bins_extremes: list, labels: list)->float:
    bins = bin_predictions(detections.copy(), bins_extremes, labels)
    # sns.barplot(data=bins, x="interval", y="match")
    plt.figure(figsize=(7, 7))
    
    bars = plt.bar(x=bins["interval"]- 0.05, height=bins["match"], width=0.09, edgecolor='black', label='Model Calibration')
    plt.bar(x=bins["interval"]- 0.05, height=bins["confidence"] - bins["match"], bottom=bins["match"], color=('red', 0.25), width=0.09, label='Calibration Gap', edgecolor=('red', 0.5))
    plt.bar_label(bars, fmt='%.2f')
    # plt.bar_label(bins["match"].to_list())
    plt.ylabel(f"Precision at IoU {iou}")
    plt.plot(bins_extremes, bins_extremes, color='red', linestyle='--')
    plt.legend()
    plt.title(f"Calibration Plot at IoU {iou} for {dataset} Dataset")
    plt.show()
    ce = compute_calibration_error(bins)
    print(f"Calibration Error at IoU {iou}: {ce:.2f}")
    return ce

def compute_iou_group(group, bboxes):
    iou_matrix = compute_iou_matrix(group,bboxes)
    return np.min(iou_matrix, axis=0)

def ensemble_merge_predictions(predictions: dict, iou_threshold: float):
    n_images = None
    for id, model_predictions in predictions.items():
        if n_images is None:
            n_images = len(model_predictions)
        else:
            assert n_images == len(model_predictions), f"All models must predict the same number of images, {id} has {len(model_predictions)} instead of {n_images}"
    merged_predictions = []
    for i in tqdm(range(n_images)):
        # Get the predictions for the i-th image
        bboxes = []
        scores = []
        for id, model_predictions in predictions.items():
            # all models have the same number of images
            bboxes.append(model_predictions[i]["boxes"])
            scores.append(model_predictions[i]["scores"])

        if len(bboxes) == 0:
            continue
        
        # Compute the IOU matrix    
        # Group the predictions based on IOU threshold
        merged_bboxes = []
        merged_scores = []
        # sort by confidence score
        bboxes = np.concatenate(bboxes)
        scores = np.concatenate(scores)
        sorted_indices = np.argsort(-scores)  # Sort indices by descending scores
        bboxes = bboxes[sorted_indices]
        scores = scores[sorted_indices]
        # print(f"Image {i}: {len(bboxes)} predictions")
        available = [True for _ in range(len(bboxes))]
        for j in range(len(bboxes)):
            # print(f"Processing box {j}/{len(bboxes)}")
            if not available[j]:
                # print(f"Box {j} already processed, skipping.")
                continue
            # Start a new group with the current box
            # print(f"Starting new group with box {j}")
            group = [j]
            bboxes_group = [bboxes[j]]
            scores_group = [scores[j]]
            available [j] = False
            while available.count(True) > 0:
                ious = compute_iou_group(bboxes_group, bboxes[available])
                k = np.argmax(ious)
                if ious[k] < iou_threshold:
                    # print(f"Group {len(merged_bboxes)}: {len(group)} boxes, IOU {ious[k]}")
                    break
                # Add the new box to the group
                # print(f"Adding box {np.where(available)[0][k]} to group")
                group.append(np.where(available)[0][k])
                bboxes_group.append(bboxes[available][k])
                scores_group.append(scores[available][k])
                available[np.where(available)[0][k]] = False
            merged_bboxes.append(np.mean(bboxes_group, axis=0)) # this could be a weighted meain
            merged_scores.append(np.sum(scores_group)/ len(predictions))
        merged_predictions.append({
            "boxes": torch.tensor(np.array(merged_bboxes)),
            "scores": torch.tensor(np.array(merged_scores)),
            "labels": torch.tensor([1 for _ in range(len(merged_bboxes))]),
            "is_crowd": torch.tensor([0 for _ in range(len(merged_bboxes))]),
            "area": torch.tensor([np.prod(box[2:] - box[:2]) for box in merged_bboxes])
        })
    return merged_predictions

def compute_map(predictions, dataset):
    map_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    labels_list = []
    for i in range(len(dataset)):
        _, labels = dataset[i]
        del labels["masks"]
        labels_list.append(labels)
    map_metric.update(predictions, labels_list)
    return map_metric.compute()