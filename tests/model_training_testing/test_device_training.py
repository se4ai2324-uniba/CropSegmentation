import pytest
import os
from sys import platform
import torch
from src.models.train_model import train
from src.config import get_global_config

BASE_PATH = '\\'.join(os.getcwd().split('\\')[:-2]) + '\\' if platform == 'win32' else '/'.join(os.getcwd().split('/')[:-2]) + '/'
config = get_global_config()
TRAIN_DATA_PATH = BASE_PATH + config.get('PROCESSED_TRAINING_DATA_PATH')
TRAIN_LABELS_PATH = BASE_PATH + config.get('PROCESSED_TRAINING_LABELS_PATH')
SAVED_MODEL_PATH = BASE_PATH + config.get('SAVED_MODEL_PATH')


def test_device_training():
	# Assert training on CPU
	loss_history_cpu, _, _ = train(
		num_epochs=1,
		ratio=.011,
		device='cpu',
		train_data_path=TRAIN_DATA_PATH,
		train_labels_path=TRAIN_LABELS_PATH,
		saved_model_path=SAVED_MODEL_PATH)
	assert loss_history_cpu

	# If CUDA is available, assert training on CUDA
	if torch.cuda.is_available():
		loss_history_cuda, _, _ = train(
			num_epochs=1,
			ratio=.012,
			device='cuda',
			train_data_path=TRAIN_DATA_PATH,
			train_labels_path=TRAIN_LABELS_PATH,
			saved_model_path=SAVED_MODEL_PATH)
		assert loss_history_cuda

	# If MPS (MacOs) is available, assert training on MPS
	if torch.backends.mps.is_available():
		loss_history_mps, _, _ = train(
			num_epochs=1,
			ratio=.013,
			device='mps',
			train_data_path=TRAIN_DATA_PATH,
			train_labels_path=TRAIN_LABELS_PATH,
			saved_model_path=SAVED_MODEL_PATH)
		assert loss_history_mps

if __name__ == "__main__":
	pytest.main()
