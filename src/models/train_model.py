import os
import sys
sys.path.append('src')
from sklearn.model_selection import train_test_split
from model import MyCustomDataset, UNet
import torch
torch.manual_seed(0)
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
from config import get_global_config
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

config = get_global_config()
TRAINING_DATA_SOURCE_PATH = config.get('PROCESSED_TRAINING_DATA_PATH')
TRAINING_LABELS_SOURCE_PATH = config.get('PROCESSED_TRAINING_LABELS_PATH')
INPUT_IMAGE_WIDTH = config.get('TILE_WIDTH')
INPUT_IMAGE_HEIGHT = config.get('TILE_HEIGHT')
SAVED_MODEL_PATH = config.get('SAVED_MODEL_PATH')
NUM_EPOCHS = config.get('NUM_EPOCHS')
BATCH_SIZE = config.get('BATCH_SIZE')
INIT_LR = config.get('INIT_LR')
RATIO = config.get('RATIO')
PARAMS_SEARCH = config.get('PARAMS_SEARCH')

if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
PIN_MEMORY = True if DEVICE == 'cuda' else False


def prepare_data():
    transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float)
                ])
    imagePaths = [TRAINING_DATA_SOURCE_PATH+i for i in os.listdir(TRAINING_DATA_SOURCE_PATH) if i != '.DS_Store']
    maskPaths = [TRAINING_LABELS_SOURCE_PATH+i for i in os.listdir(TRAINING_LABELS_SOURCE_PATH) if i != '.DS_Store']
    X_train, X_val, y_train, y_val = train_test_split(
                                                imagePaths,
                                                maskPaths,
                                                train_size = .8,
                                                test_size = .2,
                                                random_state = 3,
                                                shuffle=True)
    sampleT = int(len(X_train)*RATIO)
    sampleV = int(len(X_val)*RATIO)
    trainingDS = MyCustomDataset(
                    imagePaths=X_train[:sampleT],
                    maskPaths=y_train[:sampleT],
                    transforms=transform
                )
    validationDS = MyCustomDataset(
                    imagePaths=X_val[:sampleV],
                    maskPaths=y_val[:sampleV],
                    transforms=transform
                )
    trainLoader = DataLoader(
                    trainingDS,
                    shuffle=True,
                    batch_size=int(BATCH_SIZE),
                    pin_memory=PIN_MEMORY,
                    num_workers=os.cpu_count()
                )
    valLoader = DataLoader(
                    validationDS,
                    shuffle=False,
                    batch_size=int(BATCH_SIZE),
                    pin_memory=PIN_MEMORY,
                    num_workers=os.cpu_count()
                )
    return trainingDS, validationDS, trainLoader, valLoader

def train():
    if not os.path.isdir(SAVED_MODEL_PATH):
        Path(SAVED_MODEL_PATH).mkdir(parents=True, exist_ok=True)
    pth = str(int(NUM_EPOCHS))+'_'+str(int(BATCH_SIZE))+'_'+str(INIT_LR)+'_'+str(RATIO)+'_unet_model.pth'

    if not os.path.isfile(SAVED_MODEL_PATH+pth):
        trainingDS, validationDS, trainLoader, valLoader  = prepare_data()

        print('\n'+''.join(['> ' for i in range(25)]))
        print(f'\n{"CONFIG":<20}{"VALUE":<18}{"IS_PARAM":<18}\n')
        print(''.join(['> ' for i in range(25)]))
        print(f'{"TRAIN_SIZE":<20}{len(trainingDS):<18}{"NO":<18}')
        print(f'{"EVAL_SIZE":<20}{len(validationDS):<18}{"NO":<18}')
        print(f'{"DEVICE":<20}{DEVICE.upper():<18}{"NO":<18}')
        print(f'{"NUM_EPOCHS":<20}{int(NUM_EPOCHS):<18}{"YES":<18}')
        print(f'{"BATCH_SIZE":<20}{int(BATCH_SIZE):<18}{"YES":<18}')
        print(f'{"INIT_LR":<20}{INIT_LR:<18}{"YES":<18}')
        print(f'{"RATIO":<20}{RATIO:<18}{"YES":<18}')
        print(''.join(['> ' for i in range(25)])+'\n')

        unet = UNet(outSize=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)).to(DEVICE)
        lossFunc = BCEWithLogitsLoss()
        opt = Adam(unet.parameters(), lr=INIT_LR)
        trainSteps = len(trainingDS) // int(BATCH_SIZE)
        testSteps = len(validationDS) // int(BATCH_SIZE)
        H = {'train_loss': [], 'val_loss': [], 'test_loss': []}
        for e in tqdm(range(int(NUM_EPOCHS))):
            unet.train()
            totalTrainLoss = 0
            totalValLoss = 0
            for (i, (x, y)) in enumerate(trainLoader):
                (x, y) = (x.to(DEVICE), y.to(DEVICE))
                pred = unet(x)
                loss = lossFunc(pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                totalTrainLoss += loss
            with torch.no_grad():
                unet.eval()
                for (x, y) in valLoader:
                    (x, y) = (x.to(DEVICE), y.to(DEVICE))
                    pred = unet(x)
                    totalValLoss += lossFunc(pred, y)
                avgTrainLoss = totalTrainLoss / trainSteps
                avgValLoss = totalValLoss / testSteps
                H['train_loss'].append(avgTrainLoss.cpu().detach().numpy())
                H['val_loss'].append(avgValLoss.cpu().detach().numpy())
                print(" EPOCH: {}/{}".format(e + 1, int(NUM_EPOCHS)))
                print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgValLoss))
        torch.save(unet, SAVED_MODEL_PATH+pth)
    else:
        print('\n'+''.join(['> ' for i in range(25)]))
        print(f'\n{"WARNING: Param configuration already trained!":<35}\n')
        print(''.join(['> ' for i in range(25)])+'\n')

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        for s in PARAMS_SEARCH['BATCH_SIZE']:
            for r in PARAMS_SEARCH['INIT_LR']:
                BATCH_SIZE = s
                INIT_LR = r
                train()
    else:
        keys = [i.split('=')[0].upper() for i in args]
        values = [float(i.split('=')[1]) for i in args]
        if 'NUM_EPOCHS' in keys:
            NUM_EPOCHS = values[keys.index('NUM_EPOCHS')]
        if 'BATCH_SIZE' in keys:
            BATCH_SIZE = values[keys.index('BATCH_SIZE')]
        if 'INIT_LR' in keys:
            INIT_LR = values[keys.index('INIT_LR')]
        if 'RATIO' in keys:
            RATIO = values[keys.index('RATIO')]
        train()
