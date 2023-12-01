import pytest
import os
from sys import platform
from src.models.train_model import train
from src.config import get_global_config

BASE_PATH = '\\'.join(os.getcwd().split('\\')[:-2]) + '\\' if platform == 'win32' else '/'.join(os.getcwd().split('/')[:-2]) + '/'
config = get_global_config()
TRAIN_DATA_PATH = BASE_PATH + config.get('PROCESSED_TRAINING_DATA_PATH')
TRAIN_LABELS_PATH = BASE_PATH + config.get('PROCESSED_TRAINING_LABELS_PATH')
SAVED_MODEL_PATH = BASE_PATH + config.get('SAVED_MODEL_PATH')


def test_training_completion():
	num_epochs=5
	batch_size=config.get('BATCH_SIZE')
	init_lr=config.get('INIT_LR')
	ratio=.01
	pth = (str(int(num_epochs)) + '_' + str(int(batch_size)) +
		'_' + str(init_lr) + '_' + str(ratio)) + '_unet_model.pth'

	# Capture the model_save_path return value
	loss_history, final_lr, model_save_path = train(
		num_epochs=num_epochs,
		batch_size=batch_size,
		init_lr=init_lr,
		ratio=ratio,
		train_data_path=TRAIN_DATA_PATH,
		train_labels_path=TRAIN_LABELS_PATH,
		saved_model_path=SAVED_MODEL_PATH
	)

	# Assert that the training has likely completed by checking the final learning rate
	min_learning_rate = 0.00001  # Define a minimum learning rate threshold
	assert final_lr >= min_learning_rate

	# Assert that the loss history is not empty, implying training iterations occurred
	assert loss_history['train_loss']
	assert loss_history['val_loss']

	# New assertion to check if the model file has been saved
	assert os.path.exists(model_save_path), f"Model file not found at {model_save_path}"

	if os.path.isfile(SAVED_MODEL_PATH + pth):
		os.unlink(SAVED_MODEL_PATH + pth)


if __name__ == "__main__":
    pytest.main()
