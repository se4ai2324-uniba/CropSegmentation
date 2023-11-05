import pytest
import os
from skimage import io
from skimage.draw import rectangle
from sklearn.metrics import jaccard_score
from src.models.predict_model import make_predictions, get_saved_model

def test_directional():
    model, _ = get_saved_model()
    assert model is not False, "Saved model does not exist!"

    TESTING_DATA_SOURCE_PATH = 'data/processed/datasets_processed/testing_data/'
    TESTING_LABELS_SOURCE_PATH = 'data/processed/datasets_processed/testing_labels/'

    test_images = [TESTING_DATA_SOURCE_PATH+i for i in os.listdir(TESTING_DATA_SOURCE_PATH) if i != '.DS_Store']
    test_masks = [TESTING_LABELS_SOURCE_PATH+i for i in os.listdir(TESTING_LABELS_SOURCE_PATH) if i != '.DS_Store']

    for i,_ in enumerate(test_images):
        # Load original image
        original_img = io.imread(test_images[i])
        original_mask = io.imread(test_masks[i])

        # Create a controlled modification: Add a rectangle to the image
        rr, cc = rectangle((25, 25), extent=(50, 50), shape=original_img.shape)
        modified_img = original_img.copy()
        modified_img[rr, cc] = 255  # Assuming a white rectangle for simplicity

        modified_mask = original_mask.copy()
        modified_mask[rr, cc] = 255  # Assuming a white rectangle for simplicity

        # Make predictions for both the original and modified images
        original_pred_mask = make_predictions(model, original_img)
        modified_pred_mask = make_predictions(model, modified_img)

        j1 = jaccard_score(original_pred_mask.flatten(), original_mask.flatten(), average='micro')
        j2 = jaccard_score(modified_pred_mask.flatten(), modified_mask.flatten(), average='micro')

        assert j2 == pytest.approx(j1, abs=0.2)

if __name__ == "__main__":
    pytest.main()