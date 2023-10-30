import pytest
import torch
from src.models.train_model import prepare_data, UNet, BCEWithLogitsLoss, Adam, train_single_batch
from src.utils import getDevice
from src.config import get_global_config

DEVICE = getDevice()
config = get_global_config()
INIT_LR = config.get('INIT_LR')

@pytest.fixture
def setup_model():
    unet = UNet(outSize=(360, 480)).to(DEVICE)
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=INIT_LR)
    return unet, lossFunc, opt

def test_overfit_batch(setup_model):
    unet, lossFunc, opt = setup_model
    _, _, trainLoader, _ = prepare_data()
    x, y = next(iter(trainLoader))  # Get the first batch
    for epoch in range(10):  # Train for 10 epochs on the same batch
        _, iou = train_single_batch(unet, lossFunc, opt, x, y)
    assert iou == pytest.approx(0.95, abs=0.05)  # Check if IoU is around 0.95 ± 0.05

if __name__ == "__main__":
    pytest.main()
