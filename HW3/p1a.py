#Kyle Verdeyen
#Computer Vision EN.600.461 HW3
#p1a.py
#Performs BCE (binary entropy classification) using a siamese network.
#Operation requires 4GB+ VRAM and a NVIDIA CUDA-enabled GPU. 
#Assumes images are in ./lfw and train.txt + test.txt are in the same root directory as pyscripts
import cv2
import torch
import torchvision
import torchvision.datasets as dset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from optparse import OptionParser

#command line switches, supports -l/--load, -s/--save, -e/--epoch
parser = OptionParser()
parser.add_option("-l", "--load", action="store", type="string", dest="inputfile", help="Specify a pretrained model weight db to load")
parser.add_option("-s", "--save", action="store", type="string", dest="outputfile", help="Specify a model output filename to train the LFW dataset")
parser.add_option("-e", "--epoch", action="store", type="int", dest="epoch", default=5, help="Number of epochs, default=5")
parser.add_option("-b", "--batchsize", action="store", type="int", dest="batchsize", default=8, help="Number of batches, default=8")
(options, args) = parser.parse_args()
inputfilename = options.inputfile
outputfilename = options.outputfile
epoch = options.epoch
batchsize = options.batchsize

#dataset generator
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
			self.weights.append(z)

	#returns number of entries in dataset (number of images in inputimages1)
	def __len__(self):
		return len(self.inputimages1)

	def __getitem__(self,idx):
		image1 = torch.from_numpy(np.asarray(cv2.imread(self.inputimages1[idx], 1)))
		image2 = torch.from_numpy(np.asarray(cv2.imread(self.inputimages2[idx], 1)))
		weight = torch.from_numpy(np.asarray(self.weights[idx]).reshape[1])
		return {'image1': image1, 'image2':image2, 'weight':weight}


#P1A
#indicates we want to train a network and save it
if outputfilename is not None:
	trans = transforms.Compose([transforms.ToTensor()])
	trainingset = MakeDataset(txt_file='train.txt', root_dir="./lfw/", transform = trans)
	trainloader = DataLoader(dataset=trainingset, batch_size=batchsize, num_workers=1)
	# for image1, image2, label in trainloader:
	# 	#imagex1 = cv2.imread(image1, 1)
	# 	#imagex2 = cv2.imread(image2, 1)
	# 	cv2.imshow('test1', image1)
	# 	cv2.imshow('test2', image2)
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()
	img1, img2, weight = next(iter(trainloader))
	cv2.imshow('test', img1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#indicates we want to load neural weights and run testing
if inputfilename is not None:
	print(inputfilename) #debug
	testset = MakeDataset(txt_file='test.txt', root_dir='./lfw/')

