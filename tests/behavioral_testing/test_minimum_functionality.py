import pytest
import torch
import os
from skimage import io
from sklearn.metrics import jaccard_score
from src.models.predict_model import make_predictions, get_saved_model

best = ['00008', '00019', '00046', '00052','00053', '00076', '00099', '00110', '00136', '00145', '00181']
worst = ['00018', '00035', '00040','00059', '00063', '00069','00075','00138', '00168','00180','00193']
model_jaccard = 83.76

def test_minimum_functionality():
    model, _ = get_saved_model
    assert model is not False, "Saved model does not exist!"

    TESTING_DATA_SOURCE_PATH = 'data/processed/datasets_processed/testing_data/'
    TESTING_LABELS_SOURCE_PATH = 'data/processed/datasets_processed/testing_labels/'

    best_images = [os.path.join(TESTING_DATA_SOURCE_PATH, img, '.jpg') for img in best]
    best_masks = [os.path.join(TESTING_DATA_SOURCE_PATH, img, '.jpg') for img in best]

    jaccard_scores = []
    for i, img in enumerate(best_images):
        # Load original image
        original = io.imread(img)

        # Make predictions
        original_pred_mask = make_predictions(model, original)
        original_jaccard = jaccard_score(best_masks[i].flatten(), original_pred_mask.flatten(), average='micro')

        # Collect Jaccard scores
        jaccard_scores.append(original_jaccard)

    assert np.mean(jaccard_scores) >= model_jaccard, f"In line"