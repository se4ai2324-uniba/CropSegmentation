from locust import HttpUser, task, between
from sys import platform
import random
import os
from PIL import Image
import io
from io import BytesIO
from src.config import get_global_config
_config = get_global_config()
BASE_PATH = '\\'.join(os.getcwd().split('\\')) + '\\' if platform == 'win32' else '/'.join(os.getcwd().split('/')) + '/'
TEST_DATA_PATH = os.path.join(BASE_PATH, _config.get('PROCESSED_TESTING_DATA_PATH'))
if platform == 'win32':
    TEST_DATA_PATH = TEST_DATA_PATH.replace('/', '\\')

class CropSegmentationUser(HttpUser):
    wait_time = between(1, 5)

    @task(1)
    def main_endpoint(self):
        response = self.client.get("/")
        if response.status_code != 200:
            raise Exception("Failed to get the main endpoint")
        response_json = response.json()
        expected_keys = ['Name', 'Description', 'Version', 'Requirements', 'Github', 'DagsHub', 'Authors']
        if not all(key in response_json for key in expected_keys):
            raise Exception("The main endpoint response is missing some expected keys")

    @task(3)
    def get_samples(self):
        limit = 9
        response = self.client.get('/images?limit=' + str(limit))
        if response.status_code == 200:
            samples = response.json()['samples']
            if len(samples) != limit:
                raise Exception("The number of samples is not as expected")
            img_sample = random.choice(samples)
            response = self.client.get(f"/temp/{img_sample}")
            if response.status_code == 404:
                raise Exception(f"Image {img_sample} not found")

    @task(2)
    def upload_image(self):
        all_images = [img for img in os.listdir(TEST_DATA_PATH) if img.endswith('.jpg')]
        if not all_images:
            raise Exception("No images found in the testing data source path")
        img_name = random.choice(all_images)
        img_path = os.path.join(TEST_DATA_PATH, img_name)
        with open(img_path, 'rb') as img_file:
            image = Image.open(img_file)
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            file_content = buffer.read()
            files = {"file": (img_name, io.BytesIO(file_content).read(), "image/jpeg")}
            response = self.client.post("/upload/image", files=files)
            if response.status_code != 200:
                raise Exception(f"Image upload failed for image {img_name}")
            if response.json() != {"status": "File successfully saved!"}:
                raise Exception(f"Unexpected response for image upload {img_name}")

    @task(2)
    def predict_image(self):
        hardcoded_images = ['00045.jpg', '00183.jpg']
        for og_name in hardcoded_images:
            payload = {
                "og_name": og_name,
                "mask_name": "mask_" + og_name
            }
            response = self.client.post("/predict", json=payload)
            if response.status_code != 200:
                raise Exception(f"Prediction failed for image {og_name}")
            if "mask" not in response.json():
                raise Exception(f"No mask in the prediction response for image {og_name}")

    @task(1)
    def get_metrics(self):
        hardcoded_masks = ['00045.jpg', '00183.jpg']
        for mask_name in hardcoded_masks:
            payload = {
                "mask_name": mask_name
            }
            response = self.client.post("/metrics", json=payload)
            if response.status_code != 200:
                raise Exception(f"Metrics computation failed for mask {mask_name}")
            metrics = response.json()
            if 'truth' not in metrics or 'acc' not in metrics or 'iou' not in metrics:
                raise Exception(f"Metrics response for mask {mask_name} is incomplete")
