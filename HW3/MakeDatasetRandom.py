#Kyle Verdeyen
#Computer Vision EN.600.461 HW3
#MakeDatasetRandom.py
#Implementation of MakeDataset.py that includes random shuffling. 
#Probability of transform: 70%
#Can apply mirror flipping, rotation -/+ 30 degrees, translation -/+ 10 px, scaling 0.7-1.3
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
class MakeDatasetRandom(Dataset):
	#face training dataset
	inputimages1 = []
	inputimages2 = []
	weights = []
	def __init__(self, txt_file, root_dir, transform=None):
		self.transform = transform
		trainpath = txt_file
		trainingdata = np.genfromtxt(trainpath, dtype='str')
		for x,y,z in trainingdata:
			string1 = root_dir + x
			string2 = root_dir + y
			self.inputimages1.append(string1)
			self.inputimages2.append(string2)
			self.weights.append(int(z))

	#returns number of entries in dataset (number of entries in weights)
	def __len__(self):
		return len(self.weights)

	def __getitem__(self,idx):
		image1 = Image.open(self.inputimages1[idx])
		image2 = Image.open(self.inputimages2[idx])
		image1 = image1.convert('RGB')
		image2 = image2.convert('RGB')
		image1 = image1.resize((128, 128), Image.ANTIALIAS)
		image2 = image1.resize((128, 128), Image.ANTIALIAS)
		weight = self.weights[idx]
		if random.randint(1,11) <= 7: #70% probability of a random transformation
			#Can apply mirror flipping, rotation -/+ 30 degrees, translation -/+ 10 px, scaling 0.7-1.3
			#caveat: even after scaling image must be resized to 128*128
			#Mirror: PIL.ImageOps.mirror(image)
			#Rotate: im1.rorate(degree)
			#Translate: 
# 			x_shift = 2500
# y_shift = 1500
# a = 1
# b = 0
# c = x_shift #left/right (i.e. 5/-5)
# d = 0
# e = 1
# f = y_shift #up/down (i.e. 5/-5)
# translate = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
# # Calculate the size after cropping
# size = (translate.size[0] - x_shift, translate.size[1] - y_shift)
# # Crop to the desired size
# translate = translate.transform(size, Image.EXTENT, (0, 0, size[0], size[1]))
# translate.save('translated.tif')
			#Scale: im1.putdata(source, scale=factor)

			selector = random.randint(1,16)
			#0000 nothing doing
			if selector == 0:
				continue
			#0001 scale only

			#0010 translate only
			#0011 translate and scale
			#0100 rotate only
			#0101 rotate and scale
			#0110 rotate and translate
			#0111 rotate, translate, and scale
			#1000 mirror only
			#1001 mirror and scale
			#1010 mirror and translate
			#1011 mirror, translate, scale
			#1100 mirror, rotate
			#1101 mirror, rotate, scale
			#1110 mirror, rotate, translate
			#1111 mirror, rotate, translate, scale

		if self.transform is not None:
			image1 = self.transform(image1)
			image2 = self.transform(image2)
		return image1, image2, weight