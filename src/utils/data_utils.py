from typing import List, Tuple

import torch
import numpy as np
import cv2


def run_length_encode(mask: torch.Tensor) -> List[int]:
    """Convert a mask to RLE.
    
    Args:
        mask (torch.tensor): binary mask
        
    Returns:
        str: RLE encoded mask
    """
    # make the mask binary
    mask = (mask > 0.1).type(torch.uint8).flatten()
    # find the positions where the mask changes value
    changes = torch.diff(mask, prepend=mask[0].unsqueeze(0))
    # find the run lengths
    run_lengths = torch.where(changes != 0)[0]
    # calculate the lengths of the runs
    run_lengths = torch.diff(torch.concatenate((torch.tensor([0]), run_lengths, torch.tensor([len(mask)]))))
    return " ".join([str(i.item()) for i in run_lengths])


def run_length_decode(rle: str, shape: Tuple[int, int]) -> np.array:
    """Convert RLE to a mask.
    
    Args:
        rle (str): RLE encoded mask
        shape (Tuple[int, int]): shape of the mask
        
    Returns:
        np.ndarray: binary mask
    """
    # convert the RLE to a list of integers
    rle = list(map(int, rle.split()))
    assert sum(rle) == shape[0]*shape[1], f"Mask size ({sum(rle)}) does not match the shape ({shape[0]*shape[1]})"
    mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    starts  = rle[0::2]
    lengths = rle[1::2]
    curr_pos = 0
    for start, length in zip(starts, lengths):
        mask[curr_pos+start:curr_pos+start+length] = 1
        curr_pos += start + length
    mask = mask.reshape(shape)
    return mask


def masks_to_area(masks: List[torch.Tensor]) -> float:
    """Calculate the area of the masks.
    
    Args:
        masks (List[torch.Tensor]): list of masks
        
    Returns:
        float: area of the masks
    """
    area = 0.
    for mask in masks:
        area += mask.sum().item()
    return area

def masks_to_volume(mask: torch.Tensor) -> float:
    """Calculate the volume of the mask. We infer that the additional axes has the same lengths as the smalles on in xy.
    
    Args:
        mask (torch.Tensor): binary mask
        
    Returns:
        float: volume of the mask
    """
    area = masks_to_area(mask)
    _, _ , ax = mask_to_eccentricity(mask)
    
    if ax is  None:
        ax = (area/3.14)**0.5
    volume = 4/3*ax*area
    return volume

def masks_to_volume_2(mask: torch.Tensor) -> float:
    """Calculate the volume of the mask.
    
    Args:
        mask (torch.Tensor): binary mask
        
    Returns:
        float: volume of the mask
    """
    area = masks_to_area(mask)
    return area**1.5

def masks_to_diameter(mask: torch.Tensor) -> float:
    """Calculate the diameter of the mask.
    
    Args:
        mask (torch.Tensor): binary mask
        
    Returns:
        float: diameter of the mask
    """
    _, longax, shortax = mask_to_eccentricity(mask)
    if longax is None or shortax is None:
        return 0.
    # calculate the diameter as the average of the long and short axes
    diameter = (longax + shortax) / 2
    return diameter


def mask_to_contour(mask: torch.Tensor) -> torch.Tensor:
    """Convert a mask to a contour.
    """
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    mask = mask.cpu().numpy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # check if contours are found
    if len(contours) == 0:
        Warning("No contours found")
        return contours
    # return the largest contour
    contour = max(contours, key=cv2.contourArea)
    return contour

def mask_to_eccentricity(masks: List[torch.Tensor]) -> Tuple[float, float, float]:
    """Calculate the eccentricity of the masks.
    
    Args:
        masks (List[torch.Tensor]): list of masks
        
    Returns:
        float: eccentricity of the masks
    """
    contour = mask_to_contour(masks)    
    contour = np.array(contour).astype(np.int32)
    if len(contour) <= 5:
        return None, None, None
    ellipse = cv2.fitEllipse(contour)
    longax, shortax = max(ellipse[1]), min(ellipse[1])
    ecc = np.sqrt(1 - (shortax ** 2 / longax ** 2))
    return ecc, longax, shortax

def fill_empty_masks(masks: torch.Tensor, bboxes: torch.Tensor) -> np.ndarray:
    # fill the empty masks with the bounding boxes
    if len(masks) == 0:
        return masks
    for i, (mask, box) in enumerate(zip(masks, bboxes)):
        if mask.sum() < 5:
            x1, y1, x2, y2 = box
            # draw an ellipse in the mask
            mask = np.zeros((mask.shape[1], mask.shape[2]), dtype=np.uint8)
            mask = cv2.ellipse(mask, (int((x1+x2)/2), int((y1+y2)/2)), (int((x2-x1)/2), int((y2-y1)/2)), 0, 0, 360, 1, -1)
            # mask = cv2.GaussianBlur(mask, (5, 5), 0)
            # mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
            mask = torch.tensor(mask, dtype=torch.uint8)
            mask = mask.unsqueeze(0)
            masks[i] = mask
    return masks 