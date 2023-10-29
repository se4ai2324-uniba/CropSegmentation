"""
The U-NET model definition
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import CenterCrop
from torch.utils.data import Dataset
from skimage import io


class Block(nn.Module):
	"""
	The single block unit
	"""
	def __init__(self, inChannels, outChannels):
		super().__init__()
		self.conv1 = nn.Conv2d(inChannels, outChannels, 3)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(outChannels, outChannels, 3)
	def forward(self, x):
		"""
		The forward step
		"""
		return self.conv2(self.relu(self.conv1(x)))

class Encoder(nn.Module):
	"""
	The encoder phase of U-NET architecture
	"""
	def __init__(self, channels=(3, 16, 32, 64)):
		super().__init__()
		self.encBlocks = nn.ModuleList(
			[Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
		)
		self.pool = nn.MaxPool2d(2)
	def forward(self, x):
		"""
		The forward step
		"""
		blockOutputs = []
		for block in self.encBlocks:
			x = block(x)
			blockOutputs.append(x)
			x = self.pool(x)
		return blockOutputs

class Decoder(nn.Module):
	"""
	The decoder phase of U-NET architecture
	"""
	def __init__(self, channels=(64, 32, 16)):
		super().__init__()
		self.channels = channels
		self.upconvs = nn.ModuleList(
			[nn.ConvTranspose2d(channels[i], channels[i+1],2,2) for i in range(len(channels)-1)]
		)
		self.dec_blocks = nn.ModuleList(
			[Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
		)
	def forward(self, x, encFeatures):
		"""
		The forward step
		"""
		for i in range(len(self.channels) - 1):
			x = self.upconvs[i](x)
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)
			x = self.dec_blocks[i](x)
		return x
	def crop(self, encFeatures, x):
		"""
		Image centering and cropping
		"""
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)
		return encFeatures

class UNet(nn.Module):
	"""
	The U-NET architecture
	"""
	def __init__(self, outSize, encChannels=(3, 16, 32, 64), decChannels=(64, 32, 16), retainDim=True):
		super().__init__()
		self.encoder = Encoder(encChannels)
		self.decoder = Decoder(decChannels)
		self.head = nn.Conv2d(decChannels[-1], 1, 1)
		self.retainDim = retainDim
		self.outSize = outSize
	def forward(self, x):
		"""
		The forward step
		"""
		encFeatures = self.encoder(x)
		decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])
		myMap = self.head(decFeatures)
		if self.retainDim:
			myMap = F.interpolate(myMap, self.outSize)
		return myMap

class MyCustomDataset(Dataset):
	"""
	Custom dataset with image model preprocessing
	"""
	def __init__(self, imagePaths, maskPaths, transforms):
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
	def __len__(self):
		return len(self.imagePaths)
	def __getitem__(self, idx):
		imagePath = self.imagePaths[idx]
		maskPath = self.maskPaths[idx]
		image = io.imread(imagePath)
		mask = io.imread(maskPath)
		if self.transforms is not None:
			image = self.transforms(image)
			mask = self.transforms(mask)
		return (image, mask)
