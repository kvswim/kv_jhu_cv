#Kyle Verdeyen
#Computer Vision EN.600.461 HW3
#MakeDataset.py
#dataset generator
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
class MakeDataset(Dataset):
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
		if self.transform is not None:
			image1 = self.transform(image1)
			image2 = self.transform(image2)
		return (image1, image2, weight)