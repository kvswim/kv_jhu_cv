#Kyle Verdeyen
#kverdey1@jhu.edu
#Computer Vision EN.601.461
#Assignment 1
#Programming section 2, p6.py
#Thresholds an edge image to only return strong edges.
#Also performs Hough transform and scales to maxval=255

import cv2
import numpy as np

#loosely based on an implementation found here
# https://rosettacode.org/wiki/Hough_transform#Python
def p6(edge_image, edge_thresh): #return [edge_thresh_image, hough_image]
	#y=mx+b is not suitable
	#use xsin(theta)-ycos(theta)+rho=0
	dimensions = np.shape(edge_image)
	rows, columns = dimensions[0], dimensions[1]
	hypo = int(np.sqrt((rows**2) + (columns**2)))
	theta = 180
	rho = 800 
	diag = hypo/(rho/2) 
	edge_thresh_image = np.zeros((rows,columns))
	accumulator = np.zeros((rho, theta))
	hough_image = np.zeros((rho, theta))

	#compute hough transform
	for x in range(rows):
		for y in range(columns):
			if edge_image[x][y] > edge_thresh:
				edge_thresh_image[x][y] = 255 #over threshold, set to white
				#build accumulator
				for a in range(theta):
					temp_theta = a * (np.pi / theta)
					temp_rho = int(round(x*np.cos(temp_theta) + y*np.sin(temp_theta)) + diag)
					accumulator[temp_rho][a] += 1
			else:
				edge_thresh_image[x][y] = 0 #under threshold, set to black


	#scale hough image based on highest number of votes (max=255)
	most_votes = np.amax(accumulator)
	acc_dims = np.shape(accumulator)
	dim_rho = acc_dims[0]
	dim_theta = acc_dims[1]
	for x in range(dim_rho):
		for y in range(dim_theta):
			hough_image[x][y] = int((accumulator[x][y]/most_votes) * 255)
	return [edge_thresh_image, hough_image]