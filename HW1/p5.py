#Kyle Verdeyen
#kverdey1@jhu.edu
#Computer Vision EN.601.461
#Assignment 1
#Programming section 2, p5.py
#Finds the locations of edge points in the image. 
#this implementation employs the Sobel 3x3 gradient algorithm.

import cv2
import numpy as np
def p5(image): #return edge_image
	dimensions = np.shape(image)
	rows = dimensions[0]
	columns = dimensions[1]
	edge_image = compute_sobel_gradient(image, rows, columns)
	maximum = np.amax(edge_image)

	#now scale
	for x in np.nditer(edge_image, op_flags = ['readwrite']):
		x[...] = int((x/maximum)*255)
	return edge_image

#convolute based on sobel 3x3 mask
def compute_sobel_gradient(image, rows, columns):
	edges = np.zeros((rows, columns))
	sobelx =[[-1, 0 ,1],
	[-2, 0, 2],
	[-1, 0, 1]]
	sobely = [[1, 2, 1],
	[0, 0, 0],
	[-1, -2, -1]]
	
	#batter up!
	for x in range(rows-1):
		for y in range(columns-1):
			computed_x = (sobelx[0][0]*image[x-1][y-1]) + (sobelx[0][1]*image[x][y-1]) + (sobelx[0][2]*image[x+1][y-1]) + (sobelx[1][0]*image[x-1][y]) + (sobelx[1][1]*image[x][y]) + (sobelx[1][2]*image[x+1][y]) + (sobelx[2][0]*image[x-1][y+1]) + (sobelx[2][1]*image[x][y+1]) + (sobelx[2][2]*image[x+1][y+1])
			computed_y = (sobely[0][0]*image[x-1][y-1]) + (sobely[0][1]*image[x][y-1]) + (sobely[0][2]*image[x+1][y-1]) + (sobely[1][0]*image[x-1][y]) + (sobely[1][1]*image[x][y]) + (sobely[1][2]*image[x+1][y]) + (sobely[2][0]*image[x-1][y+1]) + (sobely[2][1]*image[x][y+1]) + (sobely[2][2]*image[x+1][y+1])
			#compute gradient magnitude
			gradient_magnitude = int(np.sqrt((computed_x**2) + (computed_y**2)))
			edges[x][y] = gradient_magnitude
	return edges