from compute_affine_xform import compute_affine_xform
from compute_proj_xform import compute_proj_xform
from detect_features import detect_features
from match_features import match_features
from nonmaxsuppts import nonmaxsuppts
from ssift_descriptor import ssift_descriptor
from display_matches import display_matches, display_ransac
from itertools import izip
import cv2
import numpy as np
import matplotlib.pyplot as plt


#iterator for image_groups
def pairwise(iterable):
	"s -> (s0, s1), (s2, s3), (s4, s5), ..."
	a = iter(iterable)
	return izip(a, a)

image_groups = ['bikes1.png', 'bikes2.png', 'bikes1.png', 'bikes3.png', 'graf1.png', 'graf2.png', 'graf1.png', 'graf3.png', 'leuven1.png', 'leuven2.png', 'leuven1.png', 'leuven3.png', 'wall1.png', 'wall2.png', 'wall1.png', 'wall3.png']

for x, y in pairwise(image_groups):
	print(x, y)
	image1 = cv2.imread(x)
	image2 = cv2.imread(y)
	dim1 = np.shape(image1)
	dim2 = np.shape(image2)
	rows1, columns1 = dim1[0], dim1[1]
	rows2, columns2 = dim2[0], dim2[1]
	x = x.strip('.png')
	y = y.strip('.png')
	features1 = detect_features(image1) #uses nonmaxsuppts
	features2 = detect_features(image2)
	matches = match_features(features1, features2, image1, image2)  #ncc descriptor computation
	sidebyside = display_matches(image1, image2, features1, features2, matches)
	titlestring = 'allmatches'+x+'and'+y+'.png'
	cv2.imshow(titlestring, sidebyside)

	#RANSAC
	affine_xform = compute_affine_xform(matches,features1, features2, image1, image2)
	ransac_sbs = display_ransac(image1, image2, features1, features2, matches, affine_xform)
	title = 'inliersOutliers'+x+'and'+y+'.png'
	cv2.imshow(titlestring,ransac_sbs)

	affine_xform = np.delete(affine_xform, (2), axis=0)
	warp1 = cv2.warpAffine(image1, affine_xform, (columns2, rows2))
	titlestring = 'warpAffine' + x + 'to' +y+'.png'
	cv2.imshow(titlestring ,warp1)