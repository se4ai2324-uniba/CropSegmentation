import numpy as np
import pytest
import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_directory, '..', '..', 'src'))
from utils import get2Dmask

def test_get2Dmask():
    # Create a dummy mask of size 100x100 with random RGB values
    dummy_mask = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    binary_mask = get2Dmask(dummy_mask)
    
    # Assert that the returned mask has a shape (100, 100)
    assert binary_mask.shape == (100, 100)
    
    # Assert that the returned mask only contains values 0 or 255
    assert np.setdiff1d(binary_mask, [0, 255]).size == 0
