import pytest
import os
from sys import platform
from src.models.train_model import prepare_data, UNet, BCEWithLogitsLoss, Adam, train_single_batch
from src.utils import getDevice
from src.config import get_global_config

BASE_PATH = '\\'.join(os.getcwd().split('\\')[:-2]) + '\\' if platform == 'win32' else '/'.join(os.getcwd().split('/')[:-2]) + '/'
config = get_global_config()
TRAIN_DATA_PATH = BASE_PATH + config.get('PROCESSED_TRAINING_DATA_PATH')
TRAIN_LABELS_PATH = BASE_PATH + config.get('PROCESSED_TRAINING_LABELS_PATH')


@pytest.fixture
def setup_model():
    unet = UNet(outSize=(360, 480)).to(getDevice())
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=config.get('INIT_LR'))
    return unet, lossFunc, opt

def test_overfit_batch(setup_model):
    unet, lossFunc, opt = setup_model
    _, _, trainLoader, _ = prepare_data(
        ratio=.02,
        train_data_path=TRAIN_DATA_PATH,
        train_labels_path=TRAIN_LABELS_PATH
    )
    x, y = next(iter(trainLoader))  # Get the first batch
    for epoch in range(100):  # Train for 50 epochs on the same batch
        _, acc = train_single_batch(unet, lossFunc, opt, x, y)
    assert acc == pytest.approx(0.89, abs=0.05)  # Check if loss is around 0.89 ± 0.05

if __name__ == "__main__":
    pytest.main()
