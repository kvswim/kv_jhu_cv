#Kyle Verdeyen
#Computer Vision EN.600.461 HW3
#p1a.py
#Performs BCE (binary entropy classification) using a siamese network
import cv2
import torch
import torchvision
import torchvision.datasets as dset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from optparse import OptionParser

#command line switches, supports -l, --load, -s, --save
parser = OptionParser()
parser.add_option("-l", "--load", action="store", type="string", dest="inputfile")
parser.add_option("-s", "--save", action="store", type="string", dest="outputfile")
(options, args) = parser.parse_args()
inputfilename = options.inputfile
outputfilename = options.outputfile


class MakeDataset(Dataset):
	#face training dataset
	def __init__(self, txt_file, root_dir, transform=None):
		trainpath = root_dir + txt_file
		trainingdata = np.genfromtxt(trainpath, dtype='str')
		inputimages1 = []
		inputimages2 = []
		weights = []
		for x,y,z in trainingdata:
			string1 = root_dir + x
			string2 = root_dir + y
			inputimages1.append(string1)
			inputimages2.append(string2)
			weights.append(z)
		self.trainingdata = trainingdata
		self.inputimages1 = inputimages1
		self.inputimages2 = inputimages2
		self.weights = weights

	def __len__(self):
		return len(self.trainingdata)

	def __getitem__(self,idx):
		image1 = cv2.imread(self.inputimages1[idx], 1)
		image2 = cv2.imread(self.inputimages2[idx], 1)
		weight = self.weights[idx]
		return {'image1': image1, 'image2':image2, 'weight':weight}


#P1A
#indicates we want to train a network and save it
if outputfilename is not None:
	trainingset = MakeDataset(txt_file='train.txt', root_dir="./lfw/")
	testset = MakeDataset(txt_file='test.txt', root_dir='./lfw/')
	

#indicates we want to load neural weights and run testing
if inputfilename is not None:
	print(inputfilename)


