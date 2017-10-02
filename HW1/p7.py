#Kyle Verdeyen
#kverdey1@jhu.edu
#Computer Vision EN.601.461
#Assignment 1
#Programming section 2, p7.py
#Finds strong lines in an image. Paints detected lines on original image.
#Hough threshold used to distinguish strong lines from short segments.

import cv2
import copy
import numpy as np
def p7(image, hough_image, hough_thresh): #return line_image
	#fancy variable assignments, bring your dress shoes
	image_dimensions, hough_dimensions = np.shape(image), np.shape(hough_image)
	image_rows, image_columns = image_dimensions[0], image_dimensions[1]
	hough_rows, hough_columns = hough_dimensions[0], hough_dimensions[1] #rho, theta
	hypo = int(np.sqrt(image_rows**2 + image_columns**2))
	diag = hypo / (hough_columns/2)
	line_image = copy.copy(image)
	for a in range(hough_rows):
		for b in range(hough_columns):
			if hough_image[a][b] > hough_thresh: #only care if edge is over threshold
				#recover theta from hough, undo what was done in p6
				temp_theta = (np.pi/hough_columns) * b
				temp_rho = int(round(a*np.cos(temp_theta) + b*np.sin(temp_theta)) + diag)
				#send to the salt mines
				vectors = linedraw(temp_theta, temp_rho)
				#draw on the original image
				cv2.line(line_image, vectors[0], vectors[1], (255, 255, 255))
	return line_image


#similar to getting 2 points in p3.getvectors()
def linedraw(theta, rho):
	x = np.sin(theta) * rho
	y = np.cos(theta) * rho
	vector1 = (np.absolute(int(x - 100 )), np.absolute(int(y - 100)))
	vector2 = (np.absolute(int(x + 100 )), np.absolute(int(y + 100)))
	return [vector1, vector2]