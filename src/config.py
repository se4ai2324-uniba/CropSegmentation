"""
A collection of global configurations
"""
CONFIG = {
	# extract_data.py
	'RAW_DATA_SOURCE_PATH':				'data/raw/datasets.zip',
	'RAW_DATA_DEST_PATH':				'data/raw/',

	# make_dataset.py
	'TILE_WIDTH':						480,
	'TILE_HEIGHT':						360,
	'RAW_TRAINING_PATH':				'data/raw/datasets/training/',
	'RAW_TESTING_PATH':					'data/raw/datasets/testing/',
	'TRAINING_FOLDERS':					[
											'Field_A/Ses1/',
											'Field_A/Ses2/',
											'Field_A/Ses3/',
											'Field_A/Ses4/',
											'Field_B/Ses1/',
											'Field_B/Ses2/'
										],
	'TESTING_FOLDERS':					[
											'weedmap/Tiles/RedEdge/003/groundtruth/',
											'weedmap/Tiles/RedEdge/003/mask/',
											'weedmap/Tiles/RedEdge/003/tile/RGB/'
										],
	'PROCESSED_TRAINING_DATA_PATH':		'data/processed/datasets_processed/training_data/',
	'PROCESSED_TRAINING_LABELS_PATH':	'data/processed/datasets_processed/training_labels/',
	'PROCESSED_TESTING_DATA_PATH':		'data/processed/datasets_processed/testing_data/',
	'PROCESSED_TESTING_LABELS_PATH':	'data/processed/datasets_processed/testing_labels/',
	'PROCESSED_ARCHIVE_PATH':			'data/processed/datasets_processed',

	# train_model.py
	'SAVED_MODEL_PATH':					'models/saved/',
	'NUM_EPOCHS':						10,
	'BATCH_SIZE':						50,
	'INIT_LR':							.001,
	'RATIO':							.25,
	'PARAMS_SEARCH':					{
											'BATCH_SIZE':	[25, 50],
											'INIT_LR':		[0.001, 0.0001, 0.0005]
										},
	'EMISSION_FILE_PATH':				'reports/carbon/emissions.csv',
	'TRACK_CO2':						0,

	# predict_model.py
	'THRESHOLD':						.5,
	'METRICS_BASE_PATH':				'models/metrics/',

	# server.py
	'BEST_MODEL_PATH':					'models/saved/sd_10_50_0.0001_0.25_unet_model.pth',
	'TEMP_PATH':						'src/api/temp/'
}


def get_global_config():
	"""
	The getter method
	"""
	return CONFIG


def set_global_config(key, value):
	"""
	The setter method
	"""
	CONFIG.update({key: value})
