"""
The preprocessing phase
"""
import os
import sys
sys.path.append('src')
from pathlib import Path
import shutil
import cv2
from utils import getTiles, applyLowLevelSegmentation, \
	applyRandomDistorsion, merge_labels
from config import get_global_config


config = get_global_config()
TILE_WIDTH = config.get('TILE_WIDTH')
TILE_HEIGHT = config.get('TILE_HEIGHT')
TRAINING_SOURCE_PATH = config.get('RAW_TRAINING_PATH')
TESTING_SOURCE_PATH = config.get('RAW_TESTING_PATH')
TRAINING_FOLDERS = config.get('TRAINING_FOLDERS')
TESTING_FOLDERS = config.get('TESTING_FOLDERS')
TRAINING_DATA_DEST_PATH = config.get('PROCESSED_TRAINING_DATA_PATH')
TESTING_DATA_DEST_PATH = config.get('PROCESSED_TESTING_DATA_PATH')
TRAINING_LABELS_DEST_PATH = config.get('PROCESSED_TRAINING_LABELS_PATH')
TESTING_LABELS_DEST_PATH = config.get('PROCESSED_TESTING_LABELS_PATH')
ARCHIVE_PATH = config.get('PROCESSED_ARCHIVE_PATH')


def training_set_preprocessing():
	"""
	Preprocess for training images
	"""
	if (
		not os.path.isdir(TRAINING_DATA_DEST_PATH) and not
		os.path.isdir(TRAINING_LABELS_DEST_PATH)
	):
		Path(TRAINING_DATA_DEST_PATH).mkdir(parents=True, exist_ok=True)
		Path(TRAINING_LABELS_DEST_PATH).mkdir(parents=True, exist_ok=True)
		for folder in TRAINING_FOLDERS:
			images_names = os.listdir(TRAINING_SOURCE_PATH+folder)
			print(f'\n{"FOLDER: " + folder:<25} \
				{"PROCESSED: " + "0/" + str(len(images_names)):<15}')
			for k, img_name in enumerate(images_names):
				if k % 10 == 0 and k != 0:
					print(f'{"":<25}'
						f'{"PROCESSED: " + str(k) + "/" + str(len(images_names)):<15}')
				if img_name != '.DS_Store':
					img = cv2.imread(TRAINING_SOURCE_PATH + folder + img_name)
					tiles = getTiles(img, H=TILE_HEIGHT, W=TILE_WIDTH)
					tiles_cluster = applyLowLevelSegmentation(tiles)
					x, y = applyRandomDistorsion(
						tiles,
						tiles_cluster,
						percentage=.2,
						maskFolder=TESTING_SOURCE_PATH + TESTING_FOLDERS[1]
					)
					for j, v in enumerate(x):
						name = str(f'{len(os.listdir(TRAINING_DATA_DEST_PATH)):05d}.jpg')
						cv2.imwrite(TRAINING_DATA_DEST_PATH + name, v)
						cv2.imwrite(TRAINING_LABELS_DEST_PATH + name, y[j])
	else:
		print('\n' + ''.join(['> ' for i in range(25)]))
		print(f'\n{"WARNING: Training folders already exist!":<35}\n')
		print(''.join(['> ' for i in range(25)]) + '\n')


def testing_set_preprocessing():
	"""
	Preprocess for testing images
	"""
	if (
		not os.path.isdir(TESTING_DATA_DEST_PATH) and not
		os.path.isdir(TESTING_LABELS_DEST_PATH)
	):
		Path(TESTING_DATA_DEST_PATH).mkdir(parents=True, exist_ok=True)
		Path(TESTING_LABELS_DEST_PATH).mkdir(parents=True, exist_ok=True)
		images_names = os.listdir(TESTING_SOURCE_PATH + TESTING_FOLDERS[2])
		for k, img_name in enumerate(images_names):
			if k % 10 == 0:
				print(f'{"TESTING DATA":<25}'
					f'{"PROCESSED: " + str(k) + "/" + str(len(images_names)):<15}')
			if img_name != '.DS_Store':
				img = cv2.imread(TESTING_SOURCE_PATH + TESTING_FOLDERS[2] + img_name)
				maskname = '003_' + img_name[:-4] + '_GroundTruth_color.png'
				mask = cv2.imread(TESTING_SOURCE_PATH + TESTING_FOLDERS[0] + maskname)
				mask = merge_labels(mask)
				name = str(f'{len(os.listdir(TESTING_DATA_DEST_PATH)):05d}.jpg')
				cv2.imwrite(TESTING_DATA_DEST_PATH + name, img)
				cv2.imwrite(TESTING_LABELS_DEST_PATH + name, mask)
	else:
		print('\n' + ''.join(['> ' for i in range(25)]))
		print(f'\n{"WARNING: Testing folders already exist!":<35}\n')
		print(''.join(['> ' for i in range(25)]) + '\n')


if __name__ == "__main__":
	training_set_preprocessing()
	testing_set_preprocessing()
	shutil.make_archive(ARCHIVE_PATH, 'zip', ARCHIVE_PATH)
