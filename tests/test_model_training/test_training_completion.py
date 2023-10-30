import pytest
import os
from src.models.train_model import train

def test_training_completion():
    loss_history, final_lr, model_save_path = train()  # Capture the model_save_path return value
    
    # Assert that the training has likely completed by checking the final learning rate
    min_learning_rate = 0.00001  # Define a minimum learning rate threshold
    assert final_lr >= min_learning_rate
    
    # Assert that the loss history is not empty, implying training iterations occurred
    assert loss_history
    
    # New assertion to check if the model file has been saved
    assert os.path.exists(model_save_path), f"Model file not found at {model_save_path}"

if __name__ == "__main__":
    pytest.main([__file__])

