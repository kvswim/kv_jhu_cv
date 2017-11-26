#Kyle Verdeyen
#Computer Vision EN.600.461 HW3
#SiameseNetworkContrastive.py
#siamese network class abstraction, without concat for BCE (used for p1b/contrastive)
import torch
import torch.nn as nn
class SiameseNetworkContrastive(nn.Module):
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


	def forward_once(self, x):
		output = self.net(x)
		output = output.view(-1, 16*16*512) #flatten layer, step 16
		output = self.vec(output)
		return output

	def forward(self, input1, input2):
		output1 = self.forward_once(input1)
		output2 = self.forward_once(input2)
		return output1, output2