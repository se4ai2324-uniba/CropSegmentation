import pytest
import os
from sys import platform
from src.models.train_model import train
from src.config import get_global_config

BASE_PATH = '\\'.join(os.getcwd().split('\\')[:-2]) + '\\' if platform == 'win32' else '/'.join(os.getcwd().split('/')[:-2]) + '/'
config = get_global_config()
TRAIN_DATA_PATH = BASE_PATH + config.get('PROCESSED_TRAINING_DATA_PATH')
TRAIN_LABELS_PATH = BASE_PATH + config.get('PROCESSED_TRAINING_LABELS_PATH')


def test_training_completion():
    # Capture the model_save_path return value
    loss_history, final_lr, model_save_path = train(
        num_epochs=5,
        ratio=.01,
        train_data_path=TRAIN_DATA_PATH,
        train_labels_path=TRAIN_LABELS_PATH
    )

    # Assert that the loss history is not empty, implying training iterations occurred
    assert loss_history['train_loss']
    assert loss_history['val_loss']

    # New assertion to check if the model file has been saved
    assert os.path.exists(model_save_path), f"Model file not found at {model_save_path}"

if __name__ == "__main__":
    pytest.main()
