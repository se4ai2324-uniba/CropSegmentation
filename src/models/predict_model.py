"""
The model prediction
"""
import os
import sys
sys.path.append('src')
import torch
import torch.multiprocessing
from torchvision import transforms
from sklearn.metrics import jaccard_score
from model import *
from skimage import io
import numpy as np
import mlflow
# import dagshub
from utils import getDevice
from config import get_global_config

torch.manual_seed(0)
torch.multiprocessing.set_sharing_strategy('file_system')

config = get_global_config()
SAVED_MODEL_PATH = config.get('SAVED_MODEL_PATH')
NUM_EPOCHS = config.get('NUM_EPOCHS')
BATCH_SIZE = config.get('BATCH_SIZE')
INIT_LR = config.get('INIT_LR')
RATIO = config.get('RATIO')
THRESHOLD = config.get('THRESHOLD')
TEST_DATA_PATH = config.get('PROCESSED_TESTING_DATA_PATH')
TEST_LABELS_PATH = config.get('PROCESSED_TESTING_LABELS_PATH')
METRICS_PATH = config.get('METRICS_BASE_PATH')
PARAMS_SEARCH = config.get('PARAMS_SEARCH')
DEVICE = getDevice()


def get_saved_model():
	"""Returns an instance of the saved model and the relative path
	Args: None
	"""
	saved_model = False
	pth = (str(int(NUM_EPOCHS)) + '_' + str(int(BATCH_SIZE)) +
		'_' + str(INIT_LR) + '_' + str(RATIO))
	pth = pth + '_unet_model.pth'
	if os.path.isfile(SAVED_MODEL_PATH + pth):
		try:
			saved_model = torch.load(
				SAVED_MODEL_PATH + pth,
				map_location=torch.device(DEVICE)
			)
		except OSError as e:
			print(e)
			sys.exit(0)
	return saved_model, pth


def get_testing_data():
	"""Returns the testing dataset with labels
	Args: None
	"""
	ko = '.DS_Store'
	# testImages contains paths
	testImages = [
		TEST_DATA_PATH + i for i in os.listdir(TEST_DATA_PATH) if i != ko
	]
	# testMasks contains BW images
	testMasks = [
		io.imread(TEST_LABELS_PATH + i) for i in os.listdir(TEST_LABELS_PATH)
		if i != ko
	]
	return testImages, testMasks


def make_predictions(model, testImgPath):
	"""Returns the predicted mask
	Args:
	model: the saved model
	testImgPath: image to segment
	"""
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
	"""Returns the pixel accuracy
	Args:
	predictions: the masks predicted by the model
	labels: the groundtruth
	"""
	all_v = predictions.shape[0] * predictions.shape[1]
	predictions = predictions.reshape(all_v)
	labels = labels.reshape(all_v)
	tp_tn = np.array([
		1 for (i, v) in enumerate(predictions)
		if v == labels[i]
	]).sum()
	return tp_tn / all_v * 100


def evaluate():
	"""Model testing function + MLFlow tracking
	Args: None
	"""
	# dagshub.init(
	# 	repo_owner='se4ai2324-uniba',
	# 	repo_name='CropSegmentation',
	# 	mlflow=True
	# )
	mlflow.start_run()
	mlflow.log_params({
		"NUM_EPOCHS":	NUM_EPOCHS,
		"BATCH_SIZE":	BATCH_SIZE,
		"INIT_LR":		INIT_LR,
		"RATIO":		RATIO
	})
	model, pth = get_saved_model()
	mlflow.set_tag("mlflow.runName", pth)
	mlflow.pytorch.log_model(model, pth)
	if model:
		testImages, testMasks = get_testing_data()
		testPred = []
		print('Making prediction...')
		for _, v in enumerate(testImages):
			pred = make_predictions(model, v)
			testPred.append(pred)
		print('Computing accuracy...')
		accuracy = {
			'unet': np.array([
				pixelAccuracy(v, testMasks[i])
				for (i, v) in enumerate(testPred)
			]),
			'unet_redu': np.array([
				pixelAccuracy(v, testMasks[i]) for (i, v)
				in enumerate(testPred)
				if not np.all(testMasks[i] == 0)
			])
		}
		print('Computing jaccard index...')
		jaccard = {
			'unet': np.array([
				jaccard_score(
					v.flatten(), testMasks[i].flatten(), average='micro'
				) * 100 for (i, v) in enumerate(testPred)
			]),
			'unet_redu': np.array([
				jaccard_score(
					v.flatten(), testMasks[i].flatten(), average='micro'
				) * 100 for (i, v) in enumerate(testPred)
				if not np.all(testMasks[i] == 0)
			])
		}
		print('RUNNED: NUM_EPOCHS: ' + str(NUM_EPOCHS))
		print('RUNNED: BATCH_SIZE: ' + str(BATCH_SIZE))
		print('RUNNED: INIT_LR: ' + str(INIT_LR))
		print('RUNNED: RATIO: ' + str(RATIO))
		print(''.join(['> ' for i in range(25)]))
		print(f'\n{"METRIC":<20}{"ALL":<18}{"NO_BLACK":<18}\n')
		print(''.join(['> ' for i in range(25)]))
		print(f'{"ACCURACY":<20}'
			f'{accuracy["unet"].mean():<18.2f}'
			f'{accuracy["unet_redu"].mean():<18.2f}')
		print(f'{"JACCARD":<20}'
			f'{jaccard["unet"].mean():<18.2f}'
			f'{jaccard["unet_redu"].mean():<18.2f}')
		print(''.join(['> ' for i in range(25)]) + '\n')
		mlflow.log_metrics(
			{
				"MEAN_AUC": accuracy["unet"].mean(),
				"MEAN_AUC_NOBLACK": accuracy["unet_redu"].mean(),
				"MEAN_IOU": jaccard["unet"].mean(),
				"MEAN_IOU_NOBLACK": jaccard["unet_redu"].mean()
			}
		)
		mlflow.end_run()
		try:
			fn = (str(NUM_EPOCHS) + '_' + str(BATCH_SIZE) +
				'_' + str(INIT_LR) + '_' + str(RATIO))
			with open(METRICS_PATH + fn + '.metrics', 'w', encoding='utf-8') as fd:
				fd.write(f'MEAN_AUC: {accuracy["unet"].mean():4f}\n')
				fd.write(f'MEAN_AUC_NOBLACK: {accuracy["unet_redu"].mean():4f}\n')
				fd.write(f'MEAN_IOU: {accuracy["unet"].mean():4f}\n')
				fd.write(f'MEAN_IOU_NOBLACK: {accuracy["unet_redu"].mean():4f}\n')
		except OSError as e:
			print(e)
	else:
		print('\n' + ''.join(['> ' for i in range(25)]))
		print(f'\n{"WARNING: Saved model does not exist!":<35}\n')
		print(''.join(['> ' for i in range(25)]) + '\n')


if __name__ == "__main__":
	args = sys.argv[1:]
	if len(args) == 0:
		for s in PARAMS_SEARCH['BATCH_SIZE']:
			for r in PARAMS_SEARCH['INIT_LR']:
				BATCH_SIZE = s
				INIT_LR = r
				evaluate()
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
		evaluate()
