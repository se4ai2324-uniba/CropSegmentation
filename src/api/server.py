"""
The API module
"""
import os
import sys
from sys import platform
if platform == 'win32':
	sys.path.append('\\'.join(os.getcwd().split('\\')[:-2])+'\src')
	sys.path.append('\\'.join(os.getcwd().split('\\')[:-2])+'\src\models')
	sys.path.append('\\'.join(os.getcwd().split('\\')[:-2])+'\src\models\model')
else:
	sys.path.append('/'.join(os.getcwd().split('/')[:-2])+'/src')
	sys.path.append('/'.join(os.getcwd().split('/')[:-2])+'/src/models')
	sys.path.append('/'.join(os.getcwd().split('/')[:-2])+'/src/models/model')
from io import BytesIO
import PIL.Image as Img
from skimage import io
from typing import Union
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import random
from pathlib import Path
from utils import getDevice, get3Dmask, get2Dmask
from models.predict_model import make_predictions, pixelAccuracy
from models.model import *
from config import get_global_config
from sklearn.metrics import jaccard_score
import torch
import cv2

torch.manual_seed(0)
torch.multiprocessing.set_sharing_strategy('file_system')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = get_global_config()
BASE_PATH = '\\'.join(os.getcwd().split('\\')[:-2]) + '\\' if platform == 'win32' else '/'.join(os.getcwd().split('/')[:-2]) + '/'
TEST_DATA_PATH = os.path.join(BASE_PATH, config.get('PROCESSED_TESTING_DATA_PATH'))
TEST_LABELS_PATH = os.path.join(BASE_PATH, config.get('PROCESSED_TESTING_LABELS_PATH'))
TEMP_PATH = config.get('TEMP_PATH')
SAVED_MODEL_PATH = os.path.join(BASE_PATH, config.get('BEST_MODEL_PATH'))
if platform == 'win32':
	TEST_DATA_PATH = TEST_DATA_PATH.replace('/', '\\')
	TEST_LABELS_PATH = TEST_LABELS_PATH.replace('/', '\\')
	TEMP_PATH = TEMP_PATH.replace('/', '\\')
	SAVED_MODEL_PATH = SAVED_MODEL_PATH.replace('/', '\\')
DEVICE = getDevice()


class Image(BaseModel):
	og_name: str | None = None
	mask_name: str | None = None


@app.get("/")
def main():
	req = {}
	file = BASE_PATH + 'requirements.txt'
	if not os.path.isfile(file):
		raise HTTPException(status_code=404, detail='File "'+file+'" not found!')
	else:
		with open(file, "r") as fp:
			for line in fp:
				para_list = line.strip().split("==")
				if len(para_list) != 2:
					continue
				req[para_list[0]] = para_list[1]
		return {
			'Name': 'Crop Segmentation',
			'Description': 'A simple tool to showcase the functionalities of the ML model.',
			'Version': '1.0.0',
			'Requirements': req,
			'Github': 'https://github.com/se4ai2324-uniba/CropSegmentation',
			'DagsHub': 'https://dagshub.com/se4ai2324-uniba/CropSegmentation',
			'Authors': ['Eleonora Ghizzota', 'Mariangela Panunzio', 'Katya Trufanova', 'Alberto G. Valerio']
		}


@app.get("/images")
def samples(limit: Union[int, None] = None):
	if not os.path.isdir(TEST_DATA_PATH):
		raise HTTPException(
			status_code=422,
			detail='Folder "'+TEST_DATA_PATH+'" not exists!'
		)
	elif False in [os.path.isfile(TEST_DATA_PATH + i) for i in os.listdir(TEST_DATA_PATH) if i != '.DS_Store']:
		raise HTTPException(
			status_code=404,
			detail='File "'+str([i for i in os.listdir(TEST_DATA_PATH) if i != '.DS_Store' and not os.path.isfile(TEST_DATA_PATH + i)])+'" not found!'
		)
	else:
		testImages = [i for i in os.listdir(TEST_DATA_PATH) if i != '.DS_Store']
		testImagesNoBlack = [i for i in testImages if io.imread(TEST_DATA_PATH + i).any()]
		sampleNames = random.sample(testImagesNoBlack, limit) if limit else testImagesNoBlack
		sampleImages = [io.imread(TEST_DATA_PATH + i) for i in sampleNames]
		if (not os.path.isdir(TEMP_PATH)):
			Path(TEMP_PATH).mkdir(parents=True, exist_ok=True)
		for i in os.listdir(TEMP_PATH):
			os.unlink(TEMP_PATH + i)
		for i, v in enumerate(sampleNames):
			io.imsave(TEMP_PATH + v, sampleImages[i])
		return {"samples": sampleNames}


@app.get("/temp/{image_name}")
def image(image_name: str):
	if image_name not in os.listdir(TEMP_PATH):
		raise HTTPException(status_code=404, detail='File "'+TEMP_PATH + image_name+'" not found!')
	else:
		return FileResponse(TEMP_PATH + image_name)


@app.post("/predict")
def predict(image: Image):
	if not os.path.isfile(SAVED_MODEL_PATH):
		raise HTTPException(status_code=422, detail='Saved model not found!')
	elif not os.path.isfile(TEMP_PATH + image.og_name):
		raise HTTPException(status_code=404, detail='File "'+TEMP_PATH + image.og_name+'" not found!')
	else:
		saved_model = torch.load(SAVED_MODEL_PATH, map_location=torch.device(DEVICE))
		orig = io.imread(TEMP_PATH + image.og_name)
		mask = make_predictions(saved_model, orig)
		colored_mask = get3Dmask(mask)
		mask_name = image.og_name.split('.')[0] + '_mask.' + image.og_name.split('.')[1]
		cv2.imwrite(TEMP_PATH + mask_name, colored_mask)
		return {"mask": mask_name}


@app.post("/upload/image")
async def upload(file: UploadFile):
	contents = await file.read()
	image = Img.open(BytesIO(contents))
	image.save('temp/upload.jpg')
	return {"status": 'File successfully saved!'}


@app.post("/metrics")
def metrics(image: Image):
	filename = ''.join(image.mask_name.split('_mask'))
	if not os.path.isfile(TEMP_PATH + image.mask_name):
		raise HTTPException(status_code=404, detail='File "'+TEMP_PATH+image.mask_name+'" not found!')
	elif not os.path.isfile(TEST_LABELS_PATH + filename):
		raise HTTPException(status_code=404, detail='File "'+TEST_LABELS_PATH+filename+'" not found!')
	else:
		pred = get2Dmask(io.imread(TEMP_PATH + image.mask_name))
		truth = io.imread(TEST_LABELS_PATH + ''.join(image.mask_name.split('_mask')))
		truth_name = filename.split('.')[0] + '_truth.' + filename.split('.')[1]
		cv2.imwrite(TEMP_PATH + truth_name, truth)
		return {
			"truth": truth_name,
			"acc": str(round(pixelAccuracy(pred, truth), 3)),
			"iou": str(round(jaccard_score(pred.flatten(), truth.flatten(), average='micro')*100, 3))
		}


if __name__ == "__main__":
	uvicorn.run(app, host="127.0.0.1", port=8000)