#Kyle Verdeyen
#Computer Vision EN.600.461 HW3
#p1a.py
#Performs BCE (binary entropy classification) using a siamese network.
#Operation requires >4GB VRAM and a NVIDIA CUDA-enabled GPU. 6GB or higher recommended. 
#Assumes images are in ./lfw and train.txt + test.txt are in the same root directory as pyscripts
import cv2
import torch
import torchvision
import time
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from optparse import OptionParser
from PIL import Image
from MakeDataset import MakeDataset
from SiameseNetwork import SiameseNetwork


#command line switches, run python p1a.py --help for info
parser = OptionParser()
parser.add_option("-l", "--load", action="store", type="string", dest="inputfile", help="Specify a pretrained model weight db to load")
parser.add_option("-s", "--save", action="store", type="string", dest="outputfile", help="Specify a model output filename to train the LFW dataset")
parser.add_option("-e", "--epoch", action="store", type="int", dest="epoch", default=5, help="Number of epochs, default=5")
parser.add_option("-b", "--batchsize", action="store", type="int", dest="batchsize", default=8, help="Number of batches, default=8")
parser.add_option("-t", "--thread", action="store", type="int", dest="numworkers", default=4, help="Number of workers to spawn in data collection, set this to number of threads available on CPU. Default=4 (quadcore no HT, eg i5)")
parser.add_option("-r", "--learnrate", action="store", type="float", dest="learnrate", default=1e-3, help="Learning rate for Adam optimizer, default=1e-3")
(options, args) = parser.parse_args()
inputfilename = options.inputfile
outputfilename = options.outputfile
epoch = options.epoch
batchsize = options.batchsize
numworkers = options.numworkers
learnrate = options.learnrate


# class Combiner(nn.Module):
# 	def __init__(self):
# 		super(Combiner, self).__init__()

# 	def forward_once(self):
		
# 	def forward(self)	

def imshow(img):
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.show()
def showplot(iteration, loss):
	plt.plot(iteration, loss)
	plt.show()

#P1A
#indicates we want to train a network and save it
if outputfilename is not None:
	trans = transforms.Compose([transforms.ToTensor()])
	trainingset = MakeDataset(txt_file='train.txt', root_dir="./lfw/", transform = trans)
	trainloader = DataLoader(dataset=trainingset, batch_size=batchsize, num_workers=numworkers)
	model = SiameseNetwork().cuda()
	criterion = nn.BCELoss().cuda()
	optimizer = optim.Adam(model.parameters(), lr=learnrate)
	itercounter=[]
	trainloss=[]
	iteration=0
	for cycle in range(epoch):
		#timestart = time.time()
		for index, data in enumerate(trainloader):
			img1, img2, weight = data
			#we initialized the model on the GPU but we need the variables too. potentially redundant
			img1 = Variable(img1, volatile=False).cuda()
			img2= Variable(img2, volatile=False).cuda()
			weight = Variable(weight, volatile=False).cuda()
			output1= model(img1, img2)
			optimizer.zero_grad()
			
			weight = weight.view(8, -1).type(torch.FloatTensor).cuda()
			# output1 = output1.type(torch.FloatTensor)
			loss = criterion(output1, weight)
			loss.backward()
			optimizer.step()
			if index % 10 == 0:
				print("Epoch {}: Current loss: {}".format(cycle, loss.data[0]))
				iteration += 10
				itercounter.append(iteration)
				trainloss.append(loss.data[0])


	model = model.cpu() #unload from gpu so we can save accurately
	showplot(itercounter, trainloss)
	torch.save(model.state_dict(), outputfilename)
	# iteration = iter(trainloader)
	# imagex1, imagex2, label = iteration.next()
	# grid = torchvision.utils.make_grid(imagex1)
	# imshow(grid)

#indicates we want to load neural weights and run testing
if inputfilename is not None:
	print(inputfilename) #debug
	trans = transforms.Compose([transforms.ToTensor()])
	testset = MakeDataset(txt_file='test.txt', root_dir='./lfw/', transform=trans)
	testloader = DataLoader(dataset=testset, batch_size=batchsize, num_workers=numworkers)
	model = SiameseNetwork().cuda()
	model.load_state_dict(torch.load(inputfilename))
	model.eval()
	everything = []
	everything_labels = []
	for index, (img1, img2, weight) in enumerate(testloader):
		#x, labels = x.cuda(), labels.cuda()
		# x = Variable(x)
		# labels = Variable(labels)
		# output = model.step(x)
		img1, img2, weight = img1.cuda(), img2.cuda(), weight.cuda() 
		img1, img2, weight = Variable(img1, requires_grad=True), Variable(img2, requires_grad=True), Variable(weight, requires_grad=True)
		output1, output2 = model.forward(img1, img2)
		everything.extend(output1.data.cpu().numpy().tolist())
		everything_labels.extend(weight.data.cpu().numpy().tolist())
	numpyall = np.array(everything)
	numpylabels = np.array(everything_labels)
	print numpyall, numpylabels
