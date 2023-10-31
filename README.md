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

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             
    │   ├── saved          <- Trained and serialized models
    │   └── metrics        <- Model predictions or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── linters        <- Report analysis of Pylint, Pynblint, Flake8, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── model.py
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   │── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │    └── visualize.py
    │   │   
    │   │── config.py      <- Global configurations
    │   └── utils.py       <- Utility functions
    │   
    ├── tests              <- Code for testing 
    |   ├── test_model_training  
    |   │   ├── test_device_training.py  
    |   │   ├── test_loss_decrease.py  
    |   │   ├── test_overfit_batch.py  
    |   │   └── test_training_completion.py  
    |   ├── behavioral_testing  
    │   │   └── test_minimum_functionality.py
    │   └── utility_testing  
    │       ├── test_applyGreenFilter.py
    │       ├── test_applyLowLevelSegmentation.py
    │       ├── test_applyRandomDistorsion.py
    │       ├── test_get2Dmask.py
    │       ├── test_getTiles.py
    │       └── test_merge_labels.py
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
