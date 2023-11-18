import os
from sys import platform
import pytest
from src.api.server import app
from fastapi.testclient import TestClient

BASE_PATH = '\\'.join(os.getcwd().split('\\')[:-2]) + '\\' if platform == 'win32' else '/'.join(os.getcwd().split('/')[:-2]) + '/'
client = TestClient(app)


def test_get_main():
    response = client.get('/')

    req = {}
    file = os.path.join(BASE_PATH, 'requirements.txt')
    with open(file, 'r') as fp:
        for line in fp:
            para_list = line.strip().split('==')
            if len(para_list) != 2:
                continue
            req[para_list[0]] = para_list[1]

    assert response.status_code == 200
    assert response.request.method == 'GET'
    assert response.json() == {'Name': 'Crop Segmentation',
			'Description': 'A simple tool to showcase the functionalities of the ML model.',
			'Version': '1.0.0',
			'Requirements': req,
			'Github': 'https://github.com/se4ai2324-uniba/CropSegmentation',
			'DagsHub': 'https://dagshub.com/se4ai2324-uniba/CropSegmentation',
			'Authors': ['Eleonora Ghizzota', 'Mariangela Panunzio', 'Katya Trufanova', 'Alberto G. Valerio']}


@pytest.mark.parametrize('limit', [2, 5, 10, 20, 0])
def test_get_samples(limit):
    response = client.get('/images?limit='+str(limit))
    assert response.status_code == 200
    assert response.request.method == 'GET'
    if limit != 0:
        assert len(response.json()['samples']) == limit
    else:
        assert len(response.json()['samples']) == 102


'''
Since the images in /temp/ are selected randomly, it is not possible to test whether a specific image is available or not.
With this test we make sure that, given wrong image names, a 404 error is returned.
'''
@pytest.mark.parametrize('image_name', ['test01.jpg', 'abcde.jpg'])
def test_get_image(image_name):
    response = client.get('/temp/{image_name}')
    assert response.status_code == 404
    assert response.request.method == "GET"


@pytest.mark.parametrize('image_name', ['00045.jpg', '00183.jpg'])
def test_post_predict(image_name):

    client.get('/images?limit=0')

    # Simulate a POST request to the /predict endpoint
    response = client.post('/predict', json={'og_name': image_name})

    # Assertions based on your expected behavior
    assert response.status_code == 200
    assert "mask" in response.json()

@pytest.mark.parametrize('mask_name', ['00045.jpg', '00183.jpg'])
def test_post_metrics(mask_name):
    data = {"mask_name": mask_name}

    # Simulate a POST request to the /metrics endpoint
    response = client.post('/metrics', json=data)

    # Assertions based on your expected behavior
    assert response.status_code == 200

    # Assuming your response contains 'truth', 'acc', and 'iou' fields
    assert 'truth' in response.json()
    assert 'acc' in response.json()
    assert 'iou' in response.json()


#def test_post_upload():