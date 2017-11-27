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
import matplotlib
matplotlib.use('Agg') #disable this if you are running locally 
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from optparse import OptionParser
from PIL import Image
from MakeDataset import MakeDataset
from SiameseNetwork import SiameseNetwork
from MakeDatasetRandom import MakeDatasetRandom


#command line switches, run python p1a.py --help for info
parser = OptionParser()
parser.add_option("-l", "--load", action="store", type="string", dest="inputfile", help="Specify a pretrained model weight db to load")
parser.add_option("-s", "--save", action="store", type="string", dest="outputfile", help="Specify a model output filename to train the LFW dataset")
parser.add_option("-e", "--epoch", action="store", type="int", dest="epoch", default=5, help="Number of epochs, default=5")
parser.add_option("-b", "--batchsize", action="store", type="int", dest="batchsize", default=8, help="Number of batches, default=8")
parser.add_option("-t", "--thread", action="store", type="int", dest="numworkers", default=4, help="Number of workers to spawn in data collection, set this to number of threads available on CPU. Default=4 (quadcore no HT, eg i5)")
parser.add_option("-r", "--learnrate", action="store", type="float", dest="learnrate", default=1e-3, help="Learning rate for Adam optimizer, default=1e-3")
parser.add_option("-a", "--random", action="store_true", dest="augmentation", default=False, help="Enable or disable data augmentation, for use with training")
(options, args) = parser.parse_args()
inputfilename = options.inputfile
outputfilename = options.outputfile
epoch = options.epoch
batchsize = options.batchsize
numworkers = options.numworkers
learnrate = options.learnrate
augment = options.augmentation

def imshow(img):
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.show()
def showplot(iteration, loss):
	plt.plot(iteration, loss)
	title = outputfilename[:-4]+"trainloss.png" #remove .pkl extension
	plt.savefig(title)
	plt.close()

#P1A
#indicates we want to train a network and save it
if outputfilename is not None:
	trans = transforms.Compose([transforms.ToTensor()])
	if not augment:
		trainingset = MakeDataset(txt_file='train.txt', root_dir="./lfw/", transform = trans)
	if augment:
		trainingset = MakeDatasetRandom(txt_file='train.txt', root_dir="./lfw/", transform = trans)
	trainloader = DataLoader(dataset=trainingset, batch_size=batchsize, num_workers=numworkers)
	model = SiameseNetwork()
	model.cuda()
	#model = model.train()
	criterion = nn.BCELoss().cuda()
	optimizer = optim.Adam(model.parameters(), lr=learnrate)
	itercounter=[]
	trainloss=[]
	iteration=0
	for cycle in range(epoch):
		for index, data in enumerate(trainloader):
			img1, img2, weight = data
			#we initialized the model on the GPU but we need the variables too. potentially redundant
			img1 = Variable(img1, volatile=False).cuda()
			img2= Variable(img2, volatile=False).cuda()
			weight = Variable(weight, volatile=False).cuda()

			output1= model(img1, img2)
			weight = weight.view(batchsize, -1).type(torch.FloatTensor).cuda() #reformat from 8 to 8x1
			loss = criterion(output1, weight)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			if index % 10 == 0: #check every 10th run per epoch 
				print("Epoch {}: Current loss: {}".format(cycle, loss.data[0]))
				iteration += 10
				itercounter.append(iteration)
				trainloss.append(loss.data[0])


	#model = model.cpu() #unload from gpu so we can save accurately
	showplot(itercounter, trainloss)
	torch.save(model.state_dict(), outputfilename)


	

#indicates we want to load neural weights and run testing
if inputfilename is not None:
	print("Loading test and train data...") #debug
	trans = transforms.Compose([transforms.ToTensor()])
	trainingset = MakeDataset(txt_file='train.txt', root_dir="./lfw/", transform = trans)
	trainloader = DataLoader(dataset=trainingset, batch_size=batchsize, num_workers=numworkers)
	
	print("Loading pretrained model from " +inputfilename)
	testmodel = SiameseNetwork()
	testmodel.load_state_dict(torch.load(inputfilename))
	testmodel.cuda()
	#model.eval()
	print("Now testing {} model vs training data:".format(inputfilename))
	errythang = []
	errythang_weights = []
	for index, data in enumerate(trainloader):
		img1, img2, weights = data
		#img1 = 
		#img2= 
		weights = Variable(weights, volatile=True).cuda()
		output1 = testmodel(Variable(img1, volatile=True).cuda(), Variable(img2, volatile=True).cuda())
		errythang.extend(output1.data.cpu().numpy().tolist())
		errythang_weights.extend(weights.data.cpu().numpy().tolist())
	numpyall = np.array(errythang)
	numpyweights = np.array(errythang_weights)
	#numpyall.reshape((numpyall.shape[0], 1))
	numpyall = numpyall[:,0]
	count=int(numpyweights.shape[0])
	correct = 0
	for x in range(0, numpyweights.shape[0]):
		if numpyall[x] > 0.5 and numpyweights[x] == 1:
			correct += 1
			
		elif numpyall[x] < 0.5 and numpyweights[x] == 0:
			correct += 1
	errorrate = float(count-correct)
	errorrate = float(errorrate/count)
	errorrate = float(errorrate*100) #have to be explicit for some reason
	print("Accuracy rate for model vs trainset (higher is better): {} %".format(errorrate))
	


	#cleanup
	del testmodel, errythang, errythang_weights, numpyall, numpyweights, count, correct, errorrate

	print("Now testing {} model vs test data:".format(inputfilename))
	testset = MakeDataset(txt_file='test.txt', root_dir='./lfw/', transform=trans)
	testloader = DataLoader(dataset=testset, batch_size=batchsize, num_workers=numworkers)
	testmodel = SiameseNetwork() #do this again just to be sure we're not getting duplicate results
	testmodel.load_state_dict(torch.load(inputfilename))
	testmodel.cuda()
	errythang = []
	errythang_weights = []
	for index, data in enumerate(testloader):
		img1, img2, weights = data
		weights = Variable(weights, volatile=True).cuda()
		output1 = testmodel(Variable(img1, volatile=True).cuda(), Variable(img2, volatile=True).cuda())
		errythang.extend(output1.data.cpu().numpy().tolist())
		errythang_weights.extend(weights.data.cpu().numpy().tolist())
	numpyall = np.array(errythang)
	numpyweights = np.array(errythang_weights)
	numpyall = numpyall[:,0]
	count=int(numpyweights.shape[0])
	correct = 0
	for x in range(0, numpyweights.shape[0]):
		if numpyall[x] > 0.5 and numpyweights[x] == 1:
			correct += 1
			
		elif numpyall[x] < 0.5 and numpyweights[x] == 0:
			correct += 1
	errorrate = float(count-correct)
	errorrate = float(errorrate/count)
	errorrate = float(errorrate*100) #have to be explicit for some reason
	print("Accuracy rate for model vs testset (higher is better): {} %".format(errorrate))