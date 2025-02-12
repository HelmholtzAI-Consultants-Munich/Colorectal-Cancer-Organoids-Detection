from typing import List, Tuple

import torch
import numpy as np


def run_length_encode(mask: torch.tensor) -> List[int]:
    """Convert a mask to RLE.
    
    Args:
        mask (torch.tensor): binary mask
        
    Returns:
        str: RLE encoded mask
    """
    # make the mask binary
    mask = (mask > 0.5).type(torch.uint8).flatten()
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