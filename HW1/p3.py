#Kyle Verdeyen
#kverdey1@jhu.edu
#Computer Vision EN.601.461
#Assignment 1
#Programming section 1, p3
#p3.py: takes labeled image and computes object attributes and generates object database
#formulas taken from Lecture 3: binary processing 
import cv2
import numpy as np
def p3(labeled_image): #return [database, output_image]
	database = {} #attributes: [label, x_position, y_position, moment, orientation, roundness]
	property_dictionary = {}  #to compute above, need [area, x_position, y_position, a, b, c, lo_theta, hi_theta, lo_E, hi_E]
	output_image = np.copy(labeled_image)
	dimensions = np.shape(labeled_image)
	rows = dimensions[0]
	columns = dimensions[1]
	#get raw area, x, y attributes for each label
	for x in range(rows):
		for y in range(columns):
			value = labeled_image[x][y] #labeled object
			if value > 0: #not a background pixel
				if value not in property_dictionary: #labeled object needs to have entry in property_dictionary
					property_dictionary[value] = [0]*10 #initialize each of the 10 properties
				property_dictionary[value][0] += 1 #area init to nonzero
				property_dictionary[value][1] += y #get x position
				property_dictionary[value][2] += x #get y position
				#prop_dict 1,2,3 all get incremented for the same label
	
	#now that we have the raw values for each label we can shift the coordinate system
	for value in property_dictionary:
		temparea, tempx, tempy = property_dictionary[value][0], property_dictionary[value][1], property_dictionary[value][2]
		property_dictionary[value][1] = tempx / temparea
		property_dictionary[value][2] = tempy / temparea 
	
	#get a, b, c necessary to compute the rest of desired properties (Lec 3 slide 15)
	for x in range(rows):
		for y in range(columns):
			value = labeled_image[x][y]
			if value > 0:
				xval, yval = property_dictionary[value][1], property_dictionary[value][2]
				xprime = x - yval
				yprime = y - xval
				property_dictionary[value][3] += (xprime**2)
				property_dictionary[value][4] += (2*xprime*yprime)
				property_dictionary[value][5] += (yprime**2)

	#compute angles (theta) and 2nd moments (E) min and max
	#for object ellipses via second moment, then arrange them into
	#the property dict according to min/max
	for value in property_dictionary:
		a, b, c = property_dictionary[value][3], property_dictionary[value][4], property_dictionary[value][5]
		diff = a - c
		theta1 = np.arctan2(b, diff)/2
		theta2 = theta1 + (np.pi/2)
		E1 = ((a*np.power(np.sin(theta1), 2))-(b*np.sin(theta1)*np.cos(theta1))+(c*np.power(np.cos(theta1),2)))
		E2 = ((a*np.power(np.sin(theta2), 2))-(b*np.sin(theta2)*np.cos(theta2))+(c*np.power(np.cos(theta2),2)))
		if E1 > E2 and theta1 > theta2: #these conditions should be mutual
			property_dictionary[value][6], property_dictionary[value][7], property_dictionary[value][8], property_dictionary[value][9] = theta2, theta1, E2, E1
		elif E2 > E1 and theta2 > theta1:
			property_dictionary[value][6], property_dictionary[value][7], property_dictionary[value][8], property_dictionary[value][9] = theta1, theta2, E1, E2
	
	#arrange database and output image
	index = 0
	for value in property_dictionary:
		label = value
		xval, yval, theta, minE, maxE = property_dictionary[value][1], property_dictionary[value][2], property_dictionary[value][6], property_dictionary[value][8], property_dictionary[value][9]
		orientation = np.degrees((np.pi/2) - theta) #calculate orientation, can extract theta later
		#orientation = np.degrees(theta - (np.pi/2))
		roundness = minE / maxE #calculate roundness
		output_properties = {'label': label, 'x_position' : xval, 'y_position' : yval, 'moment' : minE, 'orientation' : orientation, 'roundness' : roundness}
		database[index] = output_properties
		index += 1

		#draw
		cv2.circle(output_image, (xval, yval), 20, (255, 255, 255)) #image, center, radius, color (RGB)
		vector1, vector2 = getvectors(xval, yval, theta)
		cv2.line(output_image, vector1, vector2, (255, 255, 255))#image, pt1, pt2, color
	return [database, output_image]

#get a couple points to draw the line based on the minimum angle
def getvectors(xval, yval, theta):
	vector1 = (int(xval - 50 * np.cos(theta)), int(yval - 50 * np.sin(theta)))
	vector2 = (int(xval + 50 * np.cos(theta)), int(yval + 50 * np.sin(theta)))
	return vector1, vector2