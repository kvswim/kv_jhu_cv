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
from PIL import Image

#command line switches, supports -l/--load, -s/--save, -e/--epoch, -b/--batchsize, -t/--thread
parser = OptionParser()
parser.add_option("-l", "--load", action="store", type="string", dest="inputfile", help="Specify a pretrained model weight db to load")
parser.add_option("-s", "--save", action="store", type="string", dest="outputfile", help="Specify a model output filename to train the LFW dataset")
parser.add_option("-e", "--epoch", action="store", type="int", dest="epoch", default=5, help="Number of epochs, default=5")
parser.add_option("-b", "--batchsize", action="store", type="int", dest="batchsize", default=8, help="Number of batches, default=8")
parser.add_option("-t", "--thread", action="store", type="int", dest="numworkers", default=4, help="Number of workers to spawn in data collection, set this to number of threads available on CPU. Default=4 (quadcore no HT, eg i5)")
(options, args) = parser.parse_args()
inputfilename = options.inputfile
outputfilename = options.outputfile
epoch = options.epoch
batchsize = options.batchsize
numworkers = options.numworkers

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
		return image1, image2, weight

class SiameseNetwork(nn.Module):
	def __init__(self):
		super(SiameseNetwork, self).__init__()
		#steps 1-15
		self.net = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=(5,5), padding=2, stride=(1,1)),
			nn.ReLU(inplace=True), 
			nn.BatchNorm2d(64),
			nn.MaxPool2d((2,2), stride=(2,2)),
			nn.Conv2d(64, 128, kernel_size=(5,5), padding=2, stride=(1,1)),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(128),
			nn.MaxPool2d((2,2), stride=(2,2)),
			nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(256),
			nn.MaxPool2d((2,2), stride=(2,2)),
			nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(512))

		#steps 17-19
		self.vec = nn.Sequential(
			nn.Linear(16*16*512,1024),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(1024))
	def step(self, x):
		output = self.net(x)
		output = output.view(-1, 16*512) #flatten layer, step 16
		output = self.vec(output)
		return output
	def forward(self, input1, input2):
		output1 = self.step(input1)
		output2 = self.step(input2)
		return output1, output2
# def imshow(img):
# 	npimg = img.numpy()
# 	plt.imshow(np.transpose(npimg, (1,2,0)))
# 	plt.show()

#P1A
#indicates we want to train a network and save it
if outputfilename is not None:
	trans = transforms.Compose([transforms.ToTensor()])
	trainingset = MakeDataset(txt_file='train.txt', root_dir="./lfw/", transform = trans)
	trainloader = DataLoader(dataset=trainingset, batch_size=batchsize, num_workers=numworkers)
	model = SiameseNetwork().cuda()
	
	# iteration = iter(trainloader)
	# imagex1, imagex2, label = iteration.next()
	# grid = torchvision.utils.make_grid(imagex1)
	# imshow(grid)

#indicates we want to load neural weights and run testing
if inputfilename is not None:
	print(inputfilename) #debug
	testset = MakeDataset(txt_file='test.txt', root_dir='./lfw/')
	testloader = DataLoader(dataset=testset, batch_size=batchsize, num_workers=numworkers)
