import pytest
import os
import numpy as np
from skimage import io
from src.utils import applyRandomDistorsion


def test_applyRandomDistorsion(tmpdir):
    # Create a temporary directory to store dummy masks
    mask_folder = tmpdir.mkdir("masks")

    # Create some dummy masks in the temporary directory
    for i in range(5):
        dummy_mask = np.zeros((100, 100), dtype=np.uint8)
        io.imsave(os.path.join(mask_folder, f"mask_{i}.png"), dummy_mask)

    # Create a list of dummy images and masks of size 100x100
    dummy_imgs = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(10)]
    dummy_masks = [np.zeros((100, 100), dtype=np.uint8) for _ in range(10)]

    # Call the function to apply random distortions
    distorted_imgs, distorted_masks = applyRandomDistorsion(dummy_imgs, dummy_masks, 0.5, str(mask_folder))

    assert len(distorted_imgs) == 10
    assert len(distorted_masks) == 10


if __name__ == "__main__":
	pytest.main()