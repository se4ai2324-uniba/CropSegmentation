CONFIG = {
    # make_dataset.py
    'RAW_DATA_SOURCE_PATH':             'data/raw/datasets.zip',
    'RAW_DATA_DEST_PATH':               'data/raw/',

    # build_features.py
    'TILE_WIDTH':                       480,
    'TILE_HEIGHT':                      360,
    'RAW_TRAINING_PATH':                'data/raw/datasets/training/',
    'RAW_TESTING_PATH':                 'data/raw/datasets/testing/',
    'TRAINING_FOLDERS':                 [
                                            'Field_A/Ses1/',
                                            'Field_A/Ses2/',
                                            'Field_A/Ses3/',
                                            'Field_A/Ses4/',
                                            'Field_B/Ses1/',
                                            'Field_B/Ses2/'
                                        ],
    'TESTING_FOLDERS':                  [
                                            'weedmap/Tiles/RedEdge/003/groundtruth/',
                                            'weedmap/Tiles/RedEdge/003/mask/',
                                            'weedmap/Tiles/RedEdge/003/tile/RGB/'
                                        ],
    'PROCESSED_TRAINING_DATA_PATH':     'data/processed/training_data/',
    'PROCESSED_TRAINING_LABELS_PATH':   'data/processed/training_labels/',
    'PROCESSED_TESTING_DATA_PATH':      'data/processed/testing_data/',
    'PROCESSED_TESTING_LABELS_PATH':    'data/processed/testing_labels/',

    # train_model.py
    'SAVED_MODEL_PATH':                 'src/models/saved/',
    'NUM_EPOCHS':                       10,
    'BATCH_SIZE':                       50,
    'INIT_LR':                          .001,
    'RATIO':                            .25,

    # predict_model.py
    'THRESHOLD':                        .5
    
}

def get_global_config():
    return CONFIG

def set_global_config(key, value):
    CONFIG.update({key: value})