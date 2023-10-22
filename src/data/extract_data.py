import zipfile
import os
import shutil
import sys
sys.path.append('src')
from config import get_global_config

config = get_global_config()
FILE_PATH = config.get('RAW_DATA_SOURCE_PATH')
DEST_PATH = config.get('RAW_DATA_DEST_PATH')

def extract():
    
    try:
        with zipfile.ZipFile(FILE_PATH, 'r') as zip_ref:
            zip_ref.extractall(DEST_PATH)

        print('\n'+''.join(['> ' for i in range(20)]))
        print(f'\n{"SUCCESS: Files correctly extracted!":<35}\n')
        print(''.join(['> ' for i in range(20)])+'\n')

        if os.path.isdir(DEST_PATH+'/__MACOSX'):
            shutil.rmtree(DEST_PATH+'/__MACOSX')

    except Exception as e:
        print(e)
    
if __name__ == "__main__":
    extract()