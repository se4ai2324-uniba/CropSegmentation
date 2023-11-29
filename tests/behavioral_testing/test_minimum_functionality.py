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

optimal = ['00008', '00019', '00020', '00031', '00032', '00043', '00046', '00052', '00053', '00055', '00066', '00076', '00078', '00087', '00099', '00109', '00110', '00136', '00145', '00181']
challenging = ['00007', '00010', '00018', '00035', '00040','00059', '00063', '00069', '00070', '00075', '00080', '00092', '00103', '00104', '00115', '00126', '00138', '00168', '00180', '00193']

BASE_PATH = '\\'.join(os.getcwd().split('\\')[:-2]) + '\\' if platform == 'win32' else '/'.join(os.getcwd().split('/')[:-2]) + '/'
TESTING_DATA_SOURCE_PATH = BASE_PATH + 'data/processed/datasets_processed/testing_data/'
TESTING_LABELS_SOURCE_PATH = BASE_PATH + 'data/processed/datasets_processed/testing_labels/'
config = get_global_config()
SAVED_MODEL_PATH = BASE_PATH + config.get('BEST_MODEL_PATH')
DEVICE = getDevice()

model_jaccard = 0.52


def test_optimal_conditions():
    model = UNet(outSize=(360, 480)).to(DEVICE)
    model.load_state_dict(torch.load(SAVED_MODEL_PATH))
    assert model is not False, "Saved model does not exist!"
    images = [TESTING_DATA_SOURCE_PATH + img + '.jpg' for img in optimal]
    masks = [TESTING_LABELS_SOURCE_PATH + img + '.jpg' for img in optimal]

    jaccard_scores = []
    for i, img in enumerate(images):
        # Predictions
        predicted_mask = make_predictions(model, img)
        true_mask = io.imread(masks[i])
        jaccard = jaccard_score(true_mask.flatten(), predicted_mask.flatten(), average='micro')

        # Collect Jaccard scores
        jaccard_scores.append(jaccard)

    assert np.mean(jaccard_scores) >= model_jaccard, f"Model does not satify minimun functionality test"

def test_challenging_conditions():
    model = UNet(outSize=(360, 480)).to(DEVICE)
    model.load_state_dict(torch.load(SAVED_MODEL_PATH))
    assert model is not False, "Saved model does not exist!"
    images = [TESTING_DATA_SOURCE_PATH + img + '.jpg' for img in challenging]
    masks = [TESTING_LABELS_SOURCE_PATH + img + '.jpg' for img in challenging]

    jaccard_scores = []
    for i, img in enumerate(images):
        # Predictions
        predicted_mask = make_predictions(model, img)
        true_mask = io.imread(masks[i])
        jaccard = jaccard_score(true_mask.flatten(), predicted_mask.flatten(), average='micro')

        # Collect Jaccard scores
        jaccard_scores.append(jaccard)

    assert np.mean(jaccard_scores) >= model_jaccard, f"Model does not satify minimun functionality test"


if __name__ == "__main__":
    pytest.main()