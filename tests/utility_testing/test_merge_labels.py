import numpy as np
import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_directory, '..', '..', 'src'))
from utils import merge_labels

def test_merge_labels():
    # Create a dummy mask
    dummy_mask = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_mask[25:75, 25:75] = [255, 255, 255]

    merged_mask = merge_labels(dummy_mask)

    assert merged_mask.shape == (100, 100, 1)
    assert np.sum(merged_mask) == 255 * 50 * 50
