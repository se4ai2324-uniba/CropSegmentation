import pytest
import torch
from src.models.train_model import train

def test_device_training():
	# Assert training on CPU
	loss_history_cpu, _, _ = train(num_epochs=1, ratio=.011, device='cpu')
	assert loss_history_cpu

	# If CUDA is available, assert training on CUDA
	if torch.cuda.is_available():
		loss_history_cuda, _, _ = train(num_epochs=1, ratio=.012, device='cuda')
		assert loss_history_cuda

	# If MPS (MacOs) is available, assert training on MPS
	if torch.backends.mps.is_available():
		loss_history_mps, _, _ = train(num_epochs=1, ratio=.013, device='mps')
		assert loss_history_mps

if __name__ == "__main__":
	pytest.main()
