import pytest
import numpy as np
from src.utils import getTiles


def test_getTiles():
    # Create a dummy image of size 100x100
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Call the function with tile size 10x10
    tiles = getTiles(dummy_img, 10, 10)

    # Check the number of tiles
    assert len(tiles) == 100  # Since the image is 100x100 and tiles are 10x10, there should be 100 tiles

    # Check the shape of each tile
    for tile in tiles:
        assert tile.shape == (10, 10, 3)  # Each tile should be 10x10 with 3 color channels

    # Call the function with tile size 20x20
    tiles = getTiles(dummy_img, 20, 20)

    # Check the number of tiles
    assert len(tiles) == 25  # Since the image is 100x100 and tiles are 20x20, there should be 25 tiles

    # Check the shape of each tile
    for tile in tiles:
        assert tile.shape == (20, 20, 3)  # Each tile should be 20x20 with 3 color channels


if __name__ == "__main__":
	pytest.main()