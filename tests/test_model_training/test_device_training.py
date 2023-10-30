import pytest
import torch
from src.models.train_model import train

def test_device_training():
	# Assert training on CPU
	loss_history_cpu, _ = train(device=torch.device("cpu"))
	assert loss_history_cpu
	
	# If CUDA is available, assert training on CUDA
	if torch.cuda.is_available():
		loss_history_cuda, _ = train(device=torch.device("cuda"))
		assert loss_history_cuda

	# If MPS (MacOs) is available, assert training on MPS
	if torch.backends.mps.is_available():
		loss_history_mps, _ = train(device=torch.device("mps"))
		assert loss_history_mps

if __name__ == "__main__":
	pytest.main([__file__])
