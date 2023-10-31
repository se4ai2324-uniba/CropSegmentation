import cv2
import numpy as np
import pytest
import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_directory, '..', '..', 'src'))
from utils import applyGreenFilter

def test_applyGreenFilter():
    # Create a dummy image of size 100x100 with random RGB values
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    filtered_img = applyGreenFilter(dummy_img)
    
    # Assert that the returned image has the same shape as the input image
    assert filtered_img.shape == dummy_img.shape
    
    # Assert that the filtered image values are within the valid range [0, 255]
    assert (filtered_img >= 0).all() and (filtered_img <= 255).all()
