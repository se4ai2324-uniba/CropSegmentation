"""
A collection of utility functions
"""
import os
import numpy as np
import cv2
from skimage import io
import torch


def getTiles(img, H, W):
	"""Returns a set of patches of size H x W
	Args:
	img: Image to partition
	H: Height of the single tile
	W: Width of the single tile
	"""
	imgH = img.shape[0]
	imgW = img.shape[1]
	padH = int(imgH % H / 2)
	padW = int(imgW % W / 2)
	X = np.arange(0 + padH, imgH - padH, H)
	Y = np.arange(0 + padW, imgW - padW, W)
	return [img[x: x + H, y: y + W] for x in X for y in Y]


def applyLowLevelSegmentation(imgList, algorithm='clustering'):
	"""Returns a segmented image based on
	Clustering or Thresholding (aka Otsu) algorithms
	Args:
	imgList: A list of images
	algorithm: Algorithm to apply (Clustering or Thresholding)
	"""
	dummyLabels = []
	for i in imgList:
		if algorithm == 'clustering':
			_, m = clustering_segmentation(applyGreenFilter(i))
			dummyLabels.append(m)
		else:
			_, m = thresholding_segmentation(applyGreenFilter(i))
			dummyLabels.append(m)
	return dummyLabels


def applyRandomDistorsion(imgList, maskList, percentage, maskFolder):
	"""Returns a sample of distorted images by applying
	a black mask (see Weedmap dataset for details)
	Args:
	imgList: A list of images
	maskList: A list of corresponding segmentation
	percentage: Percentage of distorsion
	maskFolder: Path where are stored the masks
	"""
	masks = os.listdir(maskFolder)
	sampleSize = int(len(imgList) * percentage)
	randomId = np.random.choice(
		list(range(len(imgList))), size=sampleSize, replace=False)
	sampleMasks = np.random.choice(masks, size=sampleSize, replace=False)

	for i, filename in enumerate(sampleMasks):
		if filename != '.DS_Store':
			mask = io.imread(maskFolder + filename)
			imgList[randomId[i]] = cv2.bitwise_and(
				imgList[randomId[i]], imgList[randomId[i]], mask=mask)
			maskList[randomId[i]] = cv2.bitwise_and(
				maskList[randomId[i]], maskList[randomId[i]], mask=mask)
	return (imgList, maskList)


def merge_labels(mask):
	"""Returns the labels image in which the classes are merged in Weed/NotWeed
	Args:
	mask: Dataset provided mask
	"""
	mask_merged = np.zeros((mask.shape[0], mask.shape[1], 1), dtype='uint8')
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if mask[i][j].any():
				mask_merged[i][j] = 255
	return mask_merged


def clustering_segmentation(img):
	"""Applies image segmentation based on pixel clustering and k-Means algorithm
	Args:
	img: RGB image
	Returns:
	segmented: Segmented image in which all pixels
	are of the color of the centroid
	mask: Black and Green mask (Black=NoCrops, Green=Crops)
	"""
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
	k = 2
	img = cv2.blur(img, ksize=(15, 15))
	_, labels, (centers) = cv2.kmeans(
		np.float32(img.reshape((-1, 3))), k, None,
		criteria, 10, cv2.KMEANS_RANDOM_CENTERS
	)
	# convert all pixels to the color of the centroids
	centers = np.uint8(centers)
	labels = labels.flatten()
	segmented = centers[labels] if len(centers) else None
	segmented = segmented.reshape(img.shape)
	mask = np.copy(img)
	mask = mask.reshape((-1, 3))
	_, c = np.unique(labels, return_counts=True)
	if np.argmax(c) == 0:
		mask[labels == 1] = [0, 255, 0]
		mask[labels == 0] = [0, 0, 0]
	else:
		mask[labels == 1] = [0, 0, 0]
		mask[labels == 0] = [0, 255, 0]
		mask = mask.reshape(img.shape)
	return segmented, get2Dmask(mask)


def thresholding_segmentation(img, rgb=True, inverse=False):
	"""Applies image segmentation based on thresholding and Otsu's algorithm
	Args:
	img: Image
	rgb: Whether if the Image is RGB or not
	Returns:
	threshold: Threshold to separate pixel
	mask: Black and Green mask (Black=NoCrops, Green=Crops)
	"""
	if rgb:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	img = cv2.blur(img, ksize=(15, 15))
	# OTSU'S ALGORITHM
	hist, bins = np.histogram(img, bins=256)
	# Calculate centers of bins
	bins_mids = (bins[:-1] + bins[1:]) / 2.
	# Loop over all thresholds (indices) and get probabilities w1(t), w2(t)
	w1 = np.cumsum(hist)
	w2 = np.cumsum(hist[::-1])[::-1]
	# Get the classes means mu0(t) and mu1(t)
	mean1 = np.cumsum(hist * bins_mids) / w1
	mean2 = (np.cumsum((hist * bins_mids)[::-1]) / w2[::-1])[::-1]
	inter_class_variance = w1[:-1] * w2[1:] * (mean1[:-1] - mean2[1:]) ** 2
	index_of_max_val = np.argmax(inter_class_variance)
	threshold = bins_mids[:-1][index_of_max_val]
	_, mask = cv2.threshold(
		img, threshold, 255,
		cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
	)
	return threshold, mask


def applyGreenFilter(img):
	"""Returns a masked image in which the background
	is removed and green channel is 'highlighted'
	Args:
	img: RGB image to which apply the filter
	"""
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, (26, 10, 30), (97, 100, 255))
	return cv2.bitwise_or(img, img, mask=mask)


def get2Dmask(mask):
	"""Returns a binary mask given a RGB mask
	Args:
	mask: RGB mask
	"""
	binary_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype='uint8')
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if mask[i][j].any():
				binary_mask[i][j] = 255
	return binary_mask


def getDevice():
	"""Returns the device available on the current machine
	Args: None
	"""
	device = 'cpu'
	if torch.backends.mps.is_available():
		device = 'mps'
	elif torch.cuda.is_available():
		device = 'cuda'
	return device
