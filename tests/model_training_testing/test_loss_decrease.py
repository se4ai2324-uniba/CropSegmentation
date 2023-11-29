import pytest
import os
from sys import platform
from src.models.train_model import prepare_data, run_training_epoch, UNet
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from src.utils import getDevice
from src.config import get_global_config

BASE_PATH = '\\'.join(os.getcwd().split('\\')[:-2]) + '\\' if platform == 'win32' else '/'.join(os.getcwd().split('/')[:-2]) + '/'
config = get_global_config()
TRAIN_DATA_PATH = BASE_PATH + config.get('PROCESSED_TRAINING_DATA_PATH')
TRAIN_LABELS_PATH = BASE_PATH + config.get('PROCESSED_TRAINING_LABELS_PATH')


@pytest.fixture
def setup_training():
    _, _, trainLoader, valLoader = prepare_data(
        ratio=.02,
        train_data_path=TRAIN_DATA_PATH,
        train_labels_path=TRAIN_LABELS_PATH
    )
    unet = UNet(outSize=(360, 480)).to(getDevice())
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=config.get('INIT_LR'))
    return trainLoader, valLoader, unet, lossFunc, opt

def test_decreasing_loss(setup_training):
    trainLoader, valLoader, unet, lossFunc, opt = setup_training
    initial_loss = run_training_epoch(trainLoader, valLoader, unet, lossFunc, opt)
    subsequent_loss = run_training_epoch(trainLoader, valLoader, unet, lossFunc, opt)
    assert subsequent_loss[0] < initial_loss[0]

if __name__ == "__main__":
    pytest.main()