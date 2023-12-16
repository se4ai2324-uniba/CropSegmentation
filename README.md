Crop Segmentation
==============================

The system is designed to segment crops from the background in images collected by Unmanned Aerial Vehicles, (UAVs). It employs a Deep Neural Network (DNN) with a U-NET model, an encoder-decoder cascade structure, for semantic segmentation. The system also utilizes the K-means algorithm for further segmentation of crops in RGB color images. It is capable of processing images from different datasets and generalizing its performance across them. The system has demonstrated more accurate segmentation and convincing results compared to traditional approaches, making it a valuable tool for precision farming and sustainable agriculture.

The model has been realized for the Computer Vision exam, a.y. 2022-2023; the model card is available [here](https://github.com/se4ai2324-uniba/CropSegmentation/blob/main/models/README.md).

For further details on U-NET, read __U-Net: Convolutional Networks for Biomedical Image Segmentation__
```
@article{DBLP:journals/corr/RonnebergerFB15,
  author       = {Olaf Ronneberger and
                  Philipp Fischer and
                  Thomas Brox},
  title        = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
  journal      = {CoRR},
  volume       = {abs/1505.04597},
  year         = {2015},
  url          = {http://arxiv.org/abs/1505.04597},
  eprinttype    = {arXiv},
  eprint       = {1505.04597},
  timestamp    = {Mon, 13 Aug 2018 16:46:52 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/RonnebergerFB15.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Uptime Monitoring

We use Better Uptime to monitor the status of our website. You can check the current status and past incidents at our [Better Uptime status page](https://crop-segmentation.betteruptime.com/).


Project Organization
------------

    ├── .dvc               <- Data Version Control configurations.
    ├── .github
    │   └── workflows
    │       └── main.yaml       <- GitHub Actions workflow for the project.
    │
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    │   └── README.md      <- The data card.
    │
    ├── demo
    │   ├── css            <- The style sheets.
    │   ├── images         <- The public images folder.
    │   ├── js             <- The scripts folder.
    │   ├── Dockerfile          <- Docker file for the frontend.
    │   ├── nginx.conf          <- Configuration file for nginx.
    │   └── index.html          <- A frontend demo application.
    |
    ├── deps
    │   └── codecarbon-apple-silicon-chips    <- CodeCarbon integration.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details.
    |
    ├── log                <- CodeCarbon log.
    │
    ├── models
    │   ├── metrics        <- Model predictions or model summaries.
    │   ├── saved          <- Trained and serialized models.
    │   └── README.md      <- The model card.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, documentations and other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    |   ├── carbon         <- Generated emissions report for Carbon Emissions Tracking.
    │   ├── figures        <- Generated graphics and figures to be used in reporting.
    │   ├── linters        <- Report analysis of Pylint, Pynblint, Flake8, etc.
    |   └── locust         <- Generated reports and outputs for Load Testing with Locust.
    │
    ├── src                <- Source code for use in this project.
    │   ├── api            <- FastAPI + Uvicorn server.
    │   │   ├── redoc.py
    │   │   └── server.py
    │   │
    │   ├── data           <- Scripts to download or generate data.
    │   │   ├── extract_data.py
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling.
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions.
    │   │   ├── model.py
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   │── visualization  <- Scripts to create exploratory and results oriented visualizations.
    │   │
    │   │── config.py      <- Global configurations.
    │   └── utils.py       <- Utility functions.
    │
    ├── tests              <- Code for testing.
    │   ├── api_testing
    │   │   └── test_endpoint.py
    │   │
    │   ├── behavioral_testing
    │   │   ├── test_directional.py
    │   │   ├── test_invariance.py
    │   │   └── test_minimum_functionality.py
    │   │
    │   ├── model_training_testing
    │   │   ├── test_device_training.py
    │   │   ├── test_loss_decrease.py
    │   │   ├── test_overfit_batch.py
    │   │   └── test_training_completion.py
    │   │
    │   └── utility_testing
    │       ├── test_applyGreenFilter.py
    │       ├── test_applyLowLevelSegmentation.py
    │       ├── test_applyRandomDistorsion.py
    │       ├── test_get2Dmask.py
    │       ├── test_getTiles.py
    │       └── test_merge_labels.py
    │
    ├── .dockerignore           <- Docker ignore file.
    ├── .dvcignore              <- Data Version Control ignore file.
    ├── .flake8                 <- Flake8 ignore file.
    ├── .gitignore              <- Specifications of files to be ignored by Git.
    ├── .pylintrc               <- Configuration for Pylint.  
    ├── Dockerfile              <- Docker file for the backend.
    ├── LICENSE
    ├── Makefile                <- Makefile with commands like `make data` or `make train`.
    ├── README.md               <- The top-level README for developers using this project.
    ├── compose.yaml            <- Docker Compose configuration.
    ├── dvc.lock                <- Data Version Control record file.
    ├── dvc.yaml                <- Data Version Control pipeline file.
    ├── locustfile.py           <- Defines user behavior in load testing.
    ├── pytest.ini              <- Configuration for Pytest.
    ├── requirements-dev.txt    <- The requirements file for development environment.
    ├── requirements-prod.txt   <- The requirements file for production environment.
    ├── setup.py                <- Makes project pip installable (pip install -e .) so src can be imported.
    ├── test_environment.py     <- Checks python version.
    └── tox.ini                 <- tox file with settings for running tox; see tox.readthedocs.io.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


[https://crop-segmentation.betteruptime.com/]: https://crop-segmentation.betteruptime.com/