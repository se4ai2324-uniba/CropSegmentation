#!/usr/bin/env bash

wget -P data/raw https://cdn.albertovalerio.com/datasets/crop_segmentation/SE4AI/datasets.zip
wget -P data/processed https://cdn.albertovalerio.com/datasets/crop_segmentation/SE4AI/datasets_processed.zip
wget -P models/saved https://cdn.albertovalerio.com/datasets/crop_segmentation/SE4AI/sd_10_50_0.001_0.25_unet_model.pth

unzip data/raw/datasets.zip -d data/raw
unzip data/processed/datasets_processed.zip -d data/processed
