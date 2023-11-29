import pytest
import os
from sys import platform
import numpy as np
from skimage import io
from sklearn.metrics import jaccard_score
import torch
from src.models.predict_model import make_predictions
from src.models.model import UNet
from src.config import get_global_config
from src.utils import getDevice

BASE_PATH = '\\'.join(os.getcwd().split('\\')[:-2]) + '\\' if platform == 'win32' else '/'.join(os.getcwd().split('/')[:-2]) + '/'
TESTING_DATA_SOURCE_PATH = BASE_PATH + 'data/processed/datasets_processed/testing_data/'
TESTING_LABELS_SOURCE_PATH = BASE_PATH + 'data/processed/datasets_processed/testing_labels/'
config = get_global_config()
SAVED_MODEL_PATH = BASE_PATH + config.get('BEST_MODEL_PATH')
DEVICE = getDevice()


def test_flip_invariance():
    model = UNet(outSize=(360, 480)).to(DEVICE)
    model.load_state_dict(torch.load(SAVED_MODEL_PATH))
    assert model is not False, "Saved model does not exist!"

    test_images = [TESTING_DATA_SOURCE_PATH+i for i in os.listdir(TESTING_DATA_SOURCE_PATH) if i != '.DS_Store']
    test_masks = [io.imread(TESTING_LABELS_SOURCE_PATH+i) for i in os.listdir(TESTING_LABELS_SOURCE_PATH) if i != '.DS_Store']

    jaccard_scores = []
    for i, test_img_path in enumerate(test_images):
        # Step 1: Make a prediction for the original image and calculate the Jaccard score
        original_pred_mask = make_predictions(model, test_img_path)
        original_jaccard = jaccard_score(test_masks[i].flatten(), original_pred_mask.flatten(), average='micro')

        # step 2: Flip the image vertically
        flipped_image = np.flipud(io.imread(test_img_path))

        # step 3: make a prediction for the flipped image and calculate the Jaccard score
        flipped_pred_mask = make_predictions(model, flipped_image)
        flipped_jaccard = jaccard_score(test_masks[i].flatten(), flipped_pred_mask.flatten(), average='micro')

        # Collect the jaccard scores for the original and flipped images
        jaccard_scores.append((original_jaccard, flipped_jaccard))

    # define the tolerance for the similarity check
    tolerance = 0.5  # tolerance

    # Step 4: Check if the predictions are similar enough
    for original_jaccard, flipped_jaccard in jaccard_scores:
        assert abs(original_jaccard - flipped_jaccard) <= tolerance, f"Jaccard score difference {abs(original_jaccard - flipped_jaccard)} exceeds tolerance"


if __name__ == "__main__":
    pytest.main()