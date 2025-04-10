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
    if mask.sum() < 1:
        print("Mask is empty: ", mask.sum())
    # find the positions where the mask changes value
    changes = torch.diff(mask, prepend=mask[0].unsqueeze(0))
    # find the run lengths
    run_lengths = torch.where(changes != 0)[0]
    # calculate the lengths of the runs
    run_lengths = torch.diff(torch.concatenate((torch.tensor([0]), run_lengths, torch.tensor([len(mask)]))))
    return " ".join([str(i.item()) for i in run_lengths])


def run_length_decode(rle: List[int], shape: Tuple[int, int]) -> torch.tensor:
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

def mask_to_contour(mask: torch.Tensor) -> torch.Tensor:
    """Convert a mask to a contour.
    """
    if len(mask.shape) == 3 and mask.shape[0] == 1:
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

def mask_to_eccentricity(masks: List[torch.Tensor]) -> float:
    """Calculate the eccentricity of the masks.
    
    Args:
        masks (List[torch.Tensor]): list of masks
        
    Returns:
        float: eccentricity of the masks
    """
    contour = mask_to_contour(masks)    
    contour = np.array(contour).astype(np.int32)
    ellipse = cv2.fitEllipse(contour)
    longax, shortax = max(ellipse[1]), min(ellipse[1])
    ecc = np.sqrt(1 - (shortax ** 2 / longax ** 2))
    return ecc