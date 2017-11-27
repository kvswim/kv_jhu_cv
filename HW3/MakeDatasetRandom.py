#Kyle Verdeyen
#Computer Vision EN.600.461 HW3
#MakeDatasetRandom.py
#Implementation of MakeDataset.py that includes random shuffling. 
#Probability of transform: 70%
#Can apply mirror flipping, rotation -/+ 30 degrees, translation -/+ 10 px, scaling 0.7-1.3
import random
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
	
def randomize(image1, image2):
	if random.randint(1,11) <= 7: #70% probability of a random transformation
		#Can apply mirror flipping, rotation -/+ 30 degrees, translation -/+ 10 px, scaling 0.7-1.3 or any combination thereof
		#caveat: even after scaling image must be resized to 128*128
		#couldve done this with methods but I remebered that halfway through so it's staying like this
		selector = random.randint(1,16)
		#0000 nothing doing
		# if selector == 0:
		# 	return
		# #0001 scale only
		# if selector == 1:
		# 	# factor = random.uniform(0.7, 1.3)
		# 	# image1 = image1.putdata(image1, scale = factor)
		# 	# image2 = image2.putdata(image2, scale = factor)
		# 	return
		#0010 translate only
		if selector == 2:
			x_shift = random.randint(0,11)
			y_shift = random.randint(0,11)
			a = 1
			b = 0
			c = x_shift 
			d = 0
			e = 1
			f = y_shift 
			image1 = image1.transform(image1.size, Image.AFFINE, (a, b, c, d, e, f))
			image2 = image2.transform(image2.size, Image.AFFINE, (a, b, c, d, e, f))
		#0011 translate and scale
		if selector == 3:
			x_shift = random.randint(0,11)
			y_shift = random.randint(0,11)
			a = 1
			b = 0
			c = x_shift 
			d = 0
			e = 1
			f = y_shift 
			image1 = image1.transform(image1.size, Image.AFFINE, (a, b, c, d, e, f))
			image2 = image2.transform(image2.size, Image.AFFINE, (a, b, c, d, e, f))
			# factor = random.uniform(0.7, 1.3)
			# image1 = image1.putdata(image1, scale = factor)
			# image2 = image2.putdata(image2, scale = factor)
		#0100 rotate only
		if selector == 4:
			rotation = int(random.uniform(-30, 30))
			image1 = image1.rotate(rotation)
			image2 = image2.rotate(rotation)
		#0101 rotate and scale
		if selector == 5:
			rotation = int(random.uniform(-30, 30))
			image1 = image1.rotate(rotation)
			image2 = image2.rotate(rotation)
			# factor = random.uniform(0.7, 1.3)
			# image1 = image1.putdata(image1, scale = factor)
			# image2 = image2.putdata(image2, scale = factor)
		#0110 rotate and translate
		if selector == 6:
			rotation = int(random.uniform(-30, 30))
			image1 = image1.rotate(rotation)
			image2 = image2.rotate(rotation)
			x_shift = random.randint(0,11)
			y_shift = random.randint(0,11)
			a = 1
			b = 0
			c = x_shift 
			d = 0
			e = 1
			f = y_shift 
			image1 = image1.transform(image1.size, Image.AFFINE, (a, b, c, d, e, f))
			image2 = image2.transform(image2.size, Image.AFFINE, (a, b, c, d, e, f))
		#0111 rotate, translate, and scale
		if selector == 7:
			rotation = int(random.uniform(-30, 30))
			image1 = image1.rotate(rotation)
			image2 = image2.rotate(rotation)
			x_shift = random.randint(0,11)
			y_shift = random.randint(0,11)
			a = 1
			b = 0
			c = x_shift 
			d = 0
			e = 1
			f = y_shift 
			image1 = image1.transform(image1.size, Image.AFFINE, (a, b, c, d, e, f))
			image2 = image2.transform(image2.size, Image.AFFINE, (a, b, c, d, e, f))
			# factor = random.uniform(0.7, 1.3)
			# image1 = image1.putdata(image1, scale = factor)
			# image2 = image2.putdata(image2, scale = factor)
		#1000 mirror only
		if selector == 8:
			image1 = ImageOps.mirror(image1)
			image2 = ImageOps.mirror(image2)
		#1001 mirror and scale
		if selector == 9:
			image1 = ImageOps.mirror(image1)
			image2 = ImageOps.mirror(image2)
			# factor = random.uniform(0.7, 1.3)
			# image1 = image1.putdata(image1, scale = factor)
			# image2 = image2.putdata(image2, scale = factor)
		#1010 mirror and translate
		if selector == 10:
			image1 = ImageOps.mirror(image1)
			image2 = ImageOps.mirror(image2)
			x_shift = random.randint(0,11)
			y_shift = random.randint(0,11)
			a = 1
			b = 0
			c = x_shift 
			d = 0
			e = 1
			f = y_shift 
			image1 = image1.transform(image1.size, Image.AFFINE, (a, b, c, d, e, f))
			image2 = image2.transform(image2.size, Image.AFFINE, (a, b, c, d, e, f))
		#1011 mirror, translate, scale
		if selector == 11:
			image1 = ImageOps.mirror(image1)
			image2 = ImageOps.mirror(image2)
			x_shift = random.randint(0,11)
			y_shift = random.randint(0,11)
			a = 1
			b = 0
			c = x_shift 
			d = 0
			e = 1
			f = y_shift 
			image1 = image1.transform(image1.size, Image.AFFINE, (a, b, c, d, e, f))
			image2 = image2.transform(image2.size, Image.AFFINE, (a, b, c, d, e, f))
			# factor = random.uniform(0.7, 1.3)
			# image1 = image1.putdata(image1, scale = factor)
			# image2 = image2.putdata(image2, scale = factor)
		#1100 mirror, rotate
		if selector == 12:
			image1 = ImageOps.mirror(image1)
			image2 = ImageOps.mirror(image2)
			rotation = int(random.uniform(-30, 30))
			image1 = image1.rotate(rotation)
			image2 = image2.rotate(rotation)
		#1101 mirror, rotate, scale
		if selector == 13:
			image1 = ImageOps.mirror(image1)
			image2 = ImageOps.mirror(image2)
			rotation = int(random.uniform(-30, 30))
			image1 = image1.rotate(rotation)
			image2 = image2.rotate(rotation)
			# factor = random.uniform(0.7, 1.3)
			# image1 = image1.putdata(image1, scale = factor)
			# image2 = image2.putdata(image2, scale = factor)
		#1110 mirror, rotate, translate
		if selector == 14: 
			image1 = ImageOps.mirror(image1)
			image2 = ImageOps.mirror(image2)
			rotation = int(random.uniform(-30, 30))
			image1 = image1.rotate(rotation)
			image2 = image2.rotate(rotation)
			x_shift = random.randint(0,11)
			y_shift = random.randint(0,11)
			a = 1
			b = 0
			c = x_shift 
			d = 0
			e = 1
			f = y_shift 
			image1 = image1.transform(image1.size, Image.AFFINE, (a, b, c, d, e, f))
			image2 = image2.transform(image2.size, Image.AFFINE, (a, b, c, d, e, f))
		#1111 mirror, rotate, translate, scale
		if selector == 15:
			image1 = ImageOps.mirror(image1)
			image2 = ImageOps.mirror(image2)
			rotation = int(random.uniform(-30, 30))
			image1 = image1.rotate(rotation)
			image2 = image2.rotate(rotation)
			x_shift = random.randint(0,11)
			y_shift = random.randint(0,11)
			a = 1
			b = 0
			c = x_shift 
			d = 0
			e = 1
			f = y_shift 
			image1 = image1.transform(image1.size, Image.AFFINE, (a, b, c, d, e, f))
			image2 = image2.transform(image2.size, Image.AFFINE, (a, b, c, d, e, f))
			# factor = random.uniform(0.7, 1.3)
			# image1 = image1.putdata(image1, scale = factor)
			# image2 = image2.putdata(image2, scale = factor)

	return image1, image2


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
		
		weight = self.weights[idx]
		image1, image2 = randomize(image1, image2)
		#time to rescale
		image1 = image1.convert('RGB')
		image2 = image2.convert('RGB')
		image1 = image1.resize((128, 128), Image.ANTIALIAS)
		image2 = image1.resize((128, 128), Image.ANTIALIAS)
		if self.transform is not None:
			image1 = self.transform(image1)
			image2 = self.transform(image2)
		return (image1, image2, weight)