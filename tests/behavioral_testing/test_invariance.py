import pytest
import os
import numpy as np
from skimage import io
from sklearn.metrics import jaccard_score
from src.models.predict_model import make_predictions, get_saved_model

def test_flip_invariance():
    model, _ = get_saved_model()
    assert model is not False, "Saved model does not exist!"

    TESTING_DATA_SOURCE_PATH = 'data/processed/datasets_processed/testing_data/'
    TESTING_LABELS_SOURCE_PATH = 'data/processed/datasets_processed/testing_labels/'

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
    tolerance = 0.05  # 5% tolerance

    # Step 4: Check if the predictions are similar enough
    for original_jaccard, flipped_jaccard in jaccard_scores:
        assert abs(original_jaccard - flipped_jaccard) <= tolerance, f"Jaccard score difference {abs(original_jaccard - flipped_jaccard)} exceeds tolerance"


if __name__ == "__main__":
    pytest.main()