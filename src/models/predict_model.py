import torch
import torchvision.transforms as transforms
from sklearn.metrics import jaccard_score
import sys
sys.path.append('src')
from config import get_global_config
from skimage import io
import numpy as np
import os
from model import *

config = get_global_config()
SAVED_MODEL_PATH = config.get('SAVED_MODEL_PATH')
NUM_EPOCHS = config.get('NUM_EPOCHS')
BATCH_SIZE = config.get('BATCH_SIZE')
INIT_LR = config.get('INIT_LR')
RATIO = config.get('RATIO')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD =  config.get('THRESHOLD')
TESTING_DATA_SOURCE_PATH = config.get('PROCESSED_TESTING_DATA_PATH')
TESTING_LABELS_SOURCE_PATH = config.get('PROCESSED_TESTING_LABELS_PATH')
ACC_FILE = config.get('ACCURACY_FILE_PATH')
IOU_FILE = config.get('JACCARD_FILE_PATH')


def get_saved_model():
    saved_model = False
    pth = str(int(NUM_EPOCHS))+'_'+str(int(BATCH_SIZE))+'_'+str(INIT_LR)+'_'+str(RATIO)+'_'+str(DEVICE)+'_unet_model.pth'
    if os.path.isfile(SAVED_MODEL_PATH+pth):
        try:
            saved_model = torch.load(SAVED_MODEL_PATH+pth,  map_location=torch.device(DEVICE))
        except Exception as e:
            print(e)
            exit(0)
    return saved_model

def get_testing_data():
    # testImages contains paths
    testImages = [TESTING_DATA_SOURCE_PATH+i for i in os.listdir(TESTING_DATA_SOURCE_PATH) if i != '.DS_Store']
    # testMasks contains BW images
    testMasks = [io.imread(TESTING_LABELS_SOURCE_PATH+i) for i in os.listdir(TESTING_LABELS_SOURCE_PATH) if i != '.DS_Store']
    return testImages, testMasks

def make_predictions(model, testImgPath):
    transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float)
                ])
    image = io.imread(testImgPath)
    image = transform(image)
    image = image.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        predMask = model(image.to(DEVICE)).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()
        predMask = (predMask > THRESHOLD) * 255
        predMask = predMask.astype(np.uint8)
    return predMask

def pixelAccuracy(predictions, labels):
    all = predictions.shape[0] * predictions.shape[1]
    predictions = predictions.reshape(all)
    labels = labels.reshape(all)
    tp_tn = np.array([1 for (i,v) in enumerate(predictions) if v == labels[i]]).sum()
    return tp_tn / all * 100

def evaluate():
    model = get_saved_model()
    if model != False:
        testImages, testMasks = get_testing_data()
        testPred = []
        print('Making prediction')
        for i in range(len(testImages)):
            pred = make_predictions(model, testImages[i])
            testPred.append(pred)
        print('Computing accuracy')
        accuracy = {
            'unet': np.array([pixelAccuracy(v, testMasks[i]) for (i,v) in enumerate(testPred)]),
            'unet_redu': np.array([pixelAccuracy(v, testMasks[i]) for (i,v) in enumerate(testPred) if not np.all(testMasks[i] == 0)])
        }
        print('Computing jaccard index')
        jaccard = {
            'unet': np.array([jaccard_score(v.flatten(), testMasks[i].flatten(), average='micro')*100 for (i,v) in enumerate(testPred)]),
            'unet_redu': np.array([jaccard_score(v.flatten(), testMasks[i].flatten(), average='micro')*100 for (i,v) in enumerate(testPred) if not np.all(testMasks[i] == 0)])
        }
        print(f'\n{"the mean accuracy is: ".upper()}\033[32m \033[01m{accuracy["unet"].mean():.2f}%\033[30m \033[0m{" for the whole test set.".upper()}')
        print(f'{"the mean accuracy is: ".upper()}\033[32m \033[01m{accuracy["unet_redu"].mean():.2f}%\033[30m \033[0m{" for test set minus all black images.".upper()}')
        print(f'{"the mean IoU is: ".upper()}\033[32m \033[01m{jaccard["unet"].mean():.2f}%\033[30m \033[0m{" for the whole test set.".upper()}')
        print(f'{"the mean IoU is: ".upper()}\033[32m \033[01m{jaccard["unet_redu"].mean():.2f}%\033[30m \033[0m{" for test set minus all black images.".upper()}')
        try:
            with open(ACC_FILE, 'w') as fd:
                fd.write('MEAN AUC: {:4f}\n'.format(accuracy["unet"].mean()))
            with open(IOU_FILE, 'w') as fd:
                fd.write('MEAN IOU: {:4f}\n'.format(jaccard["unet"].mean()))
        except Exception as e:
            print(e)
    else:
        print('\n'+''.join(['> ' for i in range(25)]))
        print(f'\n{"WARNING: Saved model does not exist!":<35}\n')
        print(''.join(['> ' for i in range(25)])+'\n')


if __name__ == "__main__":
    args = sys.argv[1:]
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
    evaluate()