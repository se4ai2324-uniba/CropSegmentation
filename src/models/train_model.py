"""
The model training
"""
import os
import sys
sys.path.append('src')
from sys import platform
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import torch.multiprocessing
import numpy as np
from tqdm import tqdm
from model import MyCustomDataset, UNet
from utils import getDevice
from config import get_global_config
torch.multiprocessing.set_sharing_strategy('file_system')
torch.manual_seed(0)

BASE_PATH = '\\'.join(os.getcwd().split('\\')) + '\\' if platform == 'win32' else '/'.join(os.getcwd().split('/')) + '/'
config = get_global_config()
TRAIN_DATA_PATH = BASE_PATH + config.get('PROCESSED_TRAINING_DATA_PATH')
TRAIN_LABELS_PATH = BASE_PATH + config.get('PROCESSED_TRAINING_LABELS_PATH')
INPUT_IMAGE_WIDTH = config.get('TILE_WIDTH')
INPUT_IMAGE_HEIGHT = config.get('TILE_HEIGHT')
SAVED_MODEL_PATH = BASE_PATH + config.get('SAVED_MODEL_PATH')
NUM_EPOCHS = config.get('NUM_EPOCHS')
BATCH_SIZE = config.get('BATCH_SIZE')
INIT_LR = config.get('INIT_LR')
RATIO = config.get('RATIO')
PARAMS_SEARCH = config.get('PARAMS_SEARCH')
DEVICE = getDevice()


def compute_iou(pred, target):
	"""Returns the intesection-over-union metric
	Args:
	pred: predicted mask
	target: groundtruth
	"""
	# will need to ensure that pred and target are byte tensors
	intersection = (pred & target).float().sum((1, 2))
	union = (pred | target).float().sum((1, 2))
	# adding small value to avoid division by zero
	iou = (intersection + 1e-6) / (union + 1e-6)
	return iou.mean()


def train_single_batch(unet, lossFunc, opt, x, y):
	"""Returns the losses
	Args:
	unet: model
	lossFunc: loss function
	opt: optimizer
	x: training batch data
	y: training batch labels
	"""
	unet.train()
	(x, y) = (x.to(DEVICE), y.to(DEVICE))
	pred = unet(x)
	loss = lossFunc(pred, y)
	opt.zero_grad()
	loss.backward()
	opt.step()
	unet.eval()
	with torch.no_grad():
		predMask = pred.squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()
		# assuming 0.5 as threshold for binary segmentation
		predMask = (predMask > 0.5) * 255
		predMask = predMask.astype(np.uint8)
	y = y.cpu().numpy()
	y = y.astype(np.uint8)
	acc = accuracy_score(predMask.flatten(), y.flatten())
	return loss, acc


def prepare_data(
		batch_size=BATCH_SIZE,
		ratio=RATIO,
		train_data_path=TRAIN_DATA_PATH,
		train_labels_path=TRAIN_LABELS_PATH
	):
	"""Returns training and validation sets
	Args: None
	"""
	transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.PILToTensor(),
		transforms.ConvertImageDtype(torch.float)
	])
	ko = '.DS_Store'
	imagePaths = [
		train_data_path + i for i in os.listdir(train_data_path) if i != ko
	]
	maskPaths = [
		train_labels_path + i for i in os.listdir(train_labels_path) if i != ko
	]
	X_train, X_val, y_train, y_val = train_test_split(
		imagePaths,
		maskPaths,
		train_size=.8,
		test_size=.2,
		random_state=3,
		shuffle=True)
	sampleT = int(len(X_train) * ratio)
	sampleV = int(len(X_val) * ratio)
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
		batch_size=int(batch_size),
		pin_memory=(DEVICE == 'cuda'),
		num_workers=os.cpu_count()
	)
	valLoader = DataLoader(
		validationDS,
		shuffle=False,
		batch_size=int(batch_size),
		pin_memory=(DEVICE == 'cuda'),
		num_workers=os.cpu_count()
	)
	return trainingDS, validationDS, trainLoader, valLoader


def run_training_epoch(trainLoader, valLoader, unet, lossFunc, opt, device=DEVICE):
	"""Returns the average losses
	Args:
	trainLoader: training data
	valLoader: validation data
	unet: model
	lossFunc: loss function
	opt: optimizer
	"""
	unet.train()
	totalTrainLoss = 0
	totalValLoss = 0
	trainSteps = len(trainLoader.dataset) // len(trainLoader)
	testSteps = len(valLoader.dataset) // len(valLoader)
	for (_, (x, y)) in enumerate(trainLoader):
		(x, y) = (x.to(device), y.to(device))
		pred = unet(x)
		loss = lossFunc(pred, y)
		opt.zero_grad()
		loss.backward()
		opt.step()
		totalTrainLoss += loss
	with torch.no_grad():
		unet.eval()
		for (x, y) in valLoader:
			(x, y) = (x.to(device), y.to(device))
			pred = unet(x)
			totalValLoss += lossFunc(pred, y)
		avgTrainLoss = totalTrainLoss / trainSteps
		avgValLoss = totalValLoss / testSteps
	return avgTrainLoss, avgValLoss


def train(
		num_epochs=NUM_EPOCHS,
		batch_size=BATCH_SIZE,
		init_lr=INIT_LR,
		ratio=RATIO,
		device=DEVICE,
		train_data_path=TRAIN_DATA_PATH,
		train_labels_path=TRAIN_LABELS_PATH,
		saved_model_path=SAVED_MODEL_PATH
	):
	"""Model training function, returns the loss history,
	final learning rate and the model save path
	Args: None
	"""
	if not os.path.isdir(saved_model_path):
		Path(saved_model_path).mkdir(parents=True, exist_ok=True)
	pth = (str(int(num_epochs)) + '_' + str(int(batch_size)) + \
		'_' + str(init_lr) + '_' + str(ratio))
	pth = pth + '_unet_model.pth'
	H = {'train_loss': [], 'val_loss': [], 'test_loss': []}
	final_lr = .0
	model_save_path = os.path.join(saved_model_path, pth)
	if not os.path.isfile(saved_model_path + pth):
		trainingDS, validationDS, trainLoader, valLoader = prepare_data(
			batch_size, ratio, train_data_path, train_labels_path)

		print('\n' + ''.join(['> ' for i in range(25)]))
		print(f'\n{"CONFIG":<20}{"VALUE":<18}{"IS_PARAM":<18}\n')
		print(''.join(['> ' for i in range(25)]))
		print(f'{"TRAIN_SIZE":<20}{len(trainingDS):<18}{"NO":<18}')
		print(f'{"EVAL_SIZE":<20}{len(validationDS):<18}{"NO":<18}')
		print(f'{"DEVICE":<20}{device.upper():<18}{"NO":<18}')
		print(f'{"NUM_EPOCHS":<20}{int(num_epochs):<18}{"YES":<18}')
		print(f'{"BATCH_SIZE":<20}{int(batch_size):<18}{"YES":<18}')
		print(f'{"INIT_LR":<20}{init_lr:<18}{"YES":<18}')
		print(f'{"RATIO":<20}{ratio:<18}{"YES":<18}')
		print(''.join(['> ' for i in range(25)]) + '\n')

		unet = UNet(outSize=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)).to(device)
		lossFunc = BCEWithLogitsLoss()
		opt = Adam(unet.parameters(), lr=init_lr)

		for e in tqdm(range(int(num_epochs))):
			avgTrainLoss, avgValLoss = run_training_epoch(
				trainLoader, valLoader, unet, lossFunc, opt, device)
			H['train_loss'].append(avgTrainLoss.cpu().detach().numpy())
			H['val_loss'].append(avgValLoss.cpu().detach().numpy())
			print(f'EPOCH: {str(e+1)}/{str(int(num_epochs))}')
			print(f'Train loss: {avgTrainLoss:.6f}, Test loss: {avgValLoss:.4f}')
		torch.save(unet, model_save_path)
		final_lr = opt.param_groups[0]['lr']
	else:
		print('\n' + ''.join(['> ' for i in range(25)]))
		print(f'\n{"WARNING: Param configuration already trained!":<35}\n')
		print(''.join(['> ' for i in range(25)]) + '\n')
	return H, final_lr, model_save_path


if __name__ == "__main__":
	args = sys.argv[1:]
	if len(args) == 0:
		for s in PARAMS_SEARCH['BATCH_SIZE']:
			for r in PARAMS_SEARCH['INIT_LR']:
				train(batch_size=s, init_lr=r)
	else:
		keys = [i.split('=')[0].upper() for i in args]
		values = [float(i.split('=')[1]) for i in args]
		train(
			num_epochs=values[keys.index('NUM_EPOCHS')] if 'NUM_EPOCHS' in keys else NUM_EPOCHS,
			batch_size=values[keys.index('BATCH_SIZE')] if 'BATCH_SIZE' in keys else BATCH_SIZE,
			init_lr=values[keys.index('INIT_LR')] if 'INIT_LR' in keys else INIT_LR,
			ratio=values[keys.index('RATIO')] if 'RATIO' in keys else RATIO
		)
