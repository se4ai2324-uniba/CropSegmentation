import os
import sys
from sys import platform
sys.path.append('src')
import random
import cv2
import numpy as np
try:
	from alibi_detect.cd import KSDrift
except:
	from alibi_detect.cd import KSDrift
from config import get_global_config
from utils import get_date_time

config = get_global_config()
RATIO = config.get('RATIO')
ko = '.DS_Store'
BASE_PATH = '\\'.join(os.getcwd().split('\\')) + '\\' if platform == 'win32' else '/'.join(os.getcwd().split('/')) + '/'
TRAIN_DATA_PATH = os.path.join(BASE_PATH, config.get('PROCESSED_TRAINING_DATA_PATH'))
TEST_DATA_PATH = os.path.join(BASE_PATH, config.get('PROCESSED_TESTING_DATA_PATH'))
EMB_PATH = os.path.join(BASE_PATH, config.get('EMB_PATH'))
LOG_FILE = os.path.join(BASE_PATH, config.get('DRIFT_LOG'))
if platform == 'win32':
	TEST_DATA_PATH = TEST_DATA_PATH.replace('/', '\\')
	TRAIN_DATA_PATH = TRAIN_DATA_PATH.replace('/', '\\')
	EMB_PATH = EMB_PATH.replace('/', '\\')
	LOG_FILE = LOG_FILE.replace('/', '\\')


def get_alibi_features(
		train_data_path=TRAIN_DATA_PATH,
		test_data_path=TEST_DATA_PATH,
		ratio=RATIO
	):

	train_i = os.listdir(train_data_path)
	test_i = os.listdir(test_data_path)
	if ko in train_i: train_i.remove(ko)
	if ko in test_i: test_i.remove(ko)
	limit = int(len(train_i) * ratio)

	print('>>> (1/5) Reading training data...')
	samples_train = [cv2.imread(TRAIN_DATA_PATH + i, cv2.IMREAD_COLOR) for i in random.sample(train_i, limit)]
	grays_train = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY).flatten() for i in samples_train]
	hists_train = [np.histogram(i, 256, [0, 256])[0] for i in grays_train]

	print('>>> (2/5) Reading testing data...')
	samples_test = [cv2.imread(TEST_DATA_PATH + i, cv2.IMREAD_COLOR) for i in test_i]
	grays_test = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY).flatten() for i in samples_test]
	hists_test = [np.histogram(i, 256, [0, 256])[0] for i in grays_test]

	alpha = .6
	beta = 3

	fake = {}
	fake['imgs'] = [cv2.convertScaleAbs(img, alpha=alpha, beta=beta) for img in samples_test]
	fake['grays'] = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY).flatten() for i in fake['imgs']]
	fake['hists'] = [np.histogram(i, 256, [0, 256])[0] for i in fake['grays']]
	hists_test = fake['hists']

	print('>>> (3/5) Computing training features...')
	hist_avg_train = [i / len(hists_train) for i in list(map(sum, zip(*hists_train)))]
	print('>>> (4/5) Computing testing features...')
	hist_avg_test = [i / len(hists_test) for i in list(map(sum, zip(*hists_test)))]

	return hist_avg_train, hist_avg_test


def detect(debug=False):

	hist_avg_train, hist_avg_test = get_alibi_features()
	dt, ts = get_date_time()
	feat_files = os.listdir(EMB_PATH)

	if any('train' in file for file in feat_files):
		file = [txt for txt in feat_files if 'train' in txt]
		with open(EMB_PATH + '/' + file[0], 'r', encoding='utf-8') as f:
			old_hist_avg_train = [float(line.rstrip()) for line in f]
	else:
		with open(EMB_PATH + '/' + str(ts) + '-alibi-detect-train-features.txt', 'w', encoding='utf-8') as f:
			for value in hist_avg_train:
				f.write(f"{value}\n")
			old_hist_avg_train = hist_avg_train

	if any('test' in file for file in feat_files):
		file = [txt for txt in feat_files if 'test' in txt]
		with open(EMB_PATH + '/' + file[0], 'r', encoding='utf-8') as f:
			old_hist_avg_test = [float(line.rstrip()) for line in f]
	else:
		with open(EMB_PATH + '/' + str(ts) + '-alibi-detect-test-features.txt', 'w', encoding='utf-8') as f:
			for value in hist_avg_test:
				f.write(f"{value}\n")
			old_hist_avg_test = hist_avg_test

	print('>>> (5/5) Detecting drift...\n')
	train_drift_detector = KSDrift(np.array(old_hist_avg_train), p_val=0.01)
	train_preds_low = train_drift_detector.predict(
		np.array(hist_avg_train), drift_type='batch', return_p_val=True, return_distance=True)
	test_drift_detector = KSDrift(np.array(old_hist_avg_test), p_val=0.01)
	test_preds_low = test_drift_detector.predict(
		np.array(hist_avg_test), drift_type='batch', return_p_val=True, return_distance=True)

	is_drift = np.array([train_preds_low['data']['is_drift'], test_preds_low['data']['is_drift']])

	log = open(LOG_FILE, 'a', encoding='utf-8')
	if is_drift.any():
		if train_preds_low['data']['is_drift']:
			if debug == True:
				file = [txt for txt in feat_files if 'train' in txt]
				os.unlink(EMB_PATH + '/' + file[0])
				with open(EMB_PATH + '/' + str(ts) + '-alibi-detect-train-features.txt', 'w', encoding='utf-8') as f:
					for value in hist_avg_train:
						f.write(f"{value}\n")
			log.write('['+dt+'] ALIBI.DETECT: drift detected on training set.\n')
		if test_preds_low['data']['is_drift']:
			if debug == True:
				file = [txt for txt in feat_files if 'test' in txt]
				os.unlink(EMB_PATH + '/' + file[0])
				with open(EMB_PATH + '/' + str(ts) + '-alibi-detect-test-features.txt', 'w', encoding='utf-8') as f:
					for value in hist_avg_test:
						f.write(f"{value}\n")
			log.write('['+dt+'] ALIBI.DETECT: drift detected on testing set.\n')
	else:
		log.write('['+dt+'] ALIBI.DETECT: no drift detected.  \n')
	log.flush()
	log.close()

	return is_drift.any()


if __name__ == "__main__":
	args = sys.argv[1:]
	if len(args) == 0:
		detect()
		print('Test completed. Use "DEBUG=1" to get test output.'.upper())
	else:
		keys = [i.split('=')[0].upper() for i in args]
		values = [i.split('=')[1] for i in args]
		if 'DEBUG' in keys and values[keys.index('DEBUG')] == '1':
			if detect(debug=True):
				print('Something is seriously wrong.'.upper(), file=sys.stderr)
				sys.exit(1)
			else:
				print('Everything\'s gonna be alright.'.upper())
				sys.exit(0)