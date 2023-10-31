import pytest
import numpy as np
from src.utils import applyLowLevelSegmentation


def test_applyLowLevelSegmentation():
    # Create a list of dummy images of size 100x100
    dummy_imgs = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]

    # Call the function with algorithm as 'clustering'
    labels_clustering = applyLowLevelSegmentation(dummy_imgs, algorithm='clustering')

    # Call the function with algorithm as 'thresholding'
    labels_thresholding = applyLowLevelSegmentation(dummy_imgs, algorithm='thresholding')

    assert len(labels_clustering) == 5
    assert len(labels_thresholding) == 5


if __name__ == "__main__":
	pytest.main()