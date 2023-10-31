import pytest
from skimage import io
from sklearn.metrics import jaccard_score
from src.models.predict_model import make_predictions, get_saved_model
import numpy as np

optimal = ['00008', '00019', '00020', '00031', '00032', '00043', '00046', '00052', '00053', '00055', '00066', '00076', '00078', '00087', '00099', '00109', '00110', '00136', '00145', '00181']
challenging = ['00007', '00010', '00018', '00035', '00040','00059', '00063', '00069', '00070', '00075', '00080', '00092', '00103', '00104', '00115', '00126', '00138', '00168', '00180', '00193']

TESTING_DATA_SOURCE_PATH = 'data/processed/datasets_processed/testing_data/'
TESTING_LABELS_SOURCE_PATH = 'data/processed/datasets_processed/testing_labels/'

model_jaccard = 0.63

def test_optimal_conditions():
    model, pth = get_saved_model()
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
    model, pth = get_saved_model()
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



    
   

    


