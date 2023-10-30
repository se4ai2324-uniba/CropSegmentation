import pytest
from src.models.train_model import prepare_data, run_training_epoch, UNet
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from src.utils import getDevice
from src.config import get_global_config

DEVICE = getDevice()
config = get_global_config()
INIT_LR = config.get('INIT_LR')

@pytest.fixture
def setup_training():
    _, _, trainLoader, valLoader = prepare_data()
    unet = UNet(outSize=(360, 480)).to(DEVICE)
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=INIT_LR)
    return trainLoader, valLoader, unet, lossFunc, opt

def test_decreasing_loss(setup_training):
    trainLoader, valLoader, unet, lossFunc, opt = setup_training
    initial_loss = run_training_epoch(trainLoader, valLoader, unet, lossFunc, opt)
    subsequent_loss = run_training_epoch(trainLoader, valLoader, unet, lossFunc, opt)
    assert subsequent_loss[1] < initial_loss[1]

if __name__ == "__main__":
    pytest.main()