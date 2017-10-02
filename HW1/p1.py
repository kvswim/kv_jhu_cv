#Kyle Verdeyen
#kverdey1@jhu.edu
#Computer Vision EN.601.461
#Assignment 1
#Programming section 1, p1
#p1.py: converts a gray-level image to a binary one using a threshold value
import cv2
import numpy as np
def p1(gray_image, thresh_val): #return binary_image
	#retval, binary_out = cv2.threshold(gray_in, thresh_val, 255, cv2.THRESH_BINARY)
	binary_image = np.asarray(gray_image)
	temp = binary_image < thresh_val #boolean logical positions where image is beneath theshold
	binary_image[temp] = 0 #assign those zero
	binary_image[np.logical_not(temp)] = 255 #assign the opposite max pixel value
	return binary_image