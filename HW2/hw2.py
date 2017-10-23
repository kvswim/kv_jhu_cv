from compute_affine_xform import compute_affine_xform
from compute_proj_xform import compute_proj_xform
from detect_features import detect_features
from match_features import match_features
from nonmaxsuppts import nonmaxsuppts
from ssift_descriptor import ssift_descriptor
from itertools import izip
import cv2
import numpy as np
import matplotlib.pyplot as plt


#iterator for image_groups
def pairwise(iterable):
	"s -> (s0, s1), (s2, s3), (s4, s5), ..."
	a = iter(iterable)
	return izip(a, a)

#image_groups = ['bikes1.png', 'bikes2.png', 'bikes1.png', 'bikes3.png', 'graf1.png', 'graf2.png', 'graf1.png', 'graf3.png', 'leuven1.png', 'leuven2.png', 'leuven1.png', 'leuven3.png', 'wall1.png', 'wall2.png', 'wall1.png', 'wall3.png']
image_groups = ['bikes1.png', 'bikes2.png']
for x, y in pairwise(image_groups):
	image1 = cv2.imread(x)
	image2 = cv2.imread(y)
	features1 = detect_features(image1)
	features2 = detect_features(image2)
	matches = match_features(features1, features2, image1, image2)