#Kyle Verdeyen
#Computer Vision EN.600.461 HW3
#p1b.py
#Performs contrastive loss using a Siamese network
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