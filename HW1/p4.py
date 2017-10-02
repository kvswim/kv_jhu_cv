#Kyle Verdeyen
#kverdey1@jhu.edu
#Computer Vision EN.601.461
#Assignment 1
#Programming section 1, p4
#Recognizes objects in an image given a database.
import cv2
import numpy as np
from p3 import p3, getvectors
def p4(labeled_image, database): #return output_image
	output_image = np.copy(labeled_image)
	image_db, throwaway = p3(labeled_image)
	recognized_db = []
	image_dimension = len(image_db) #number of total objects in image to analyze
	reference_dimension = len(database) #number of objects in reference database
	
	#rudimentary brute force object recognition based on property similarity between objects in test image
	#and objects in database. If moment and roundness have >80% similarity they are 
	#considered recognized. an actual algorithm like sift would be nice but I'm kinda dumb
	for x in range(image_dimension):
		moment = image_db[x]['moment']
		roundness = image_db[x]['roundness']
		for y in range(reference_dimension): #
			obj_moment = database[y]['moment']
			obj_roundness = database[y]['roundness']
			#minimizing possible ratio (attempt to keep similarity calculation as a percentage 0-1)
			moment_ratio = min((obj_moment/moment), (moment/obj_moment))
			roundness_ratio = min((obj_roundness/roundness), (roundness/obj_roundness))
			if moment_ratio > .8 and roundness_ratio > .8: #still greater than 80%? just assume it's interesting
				recognized_db.append(image_db[x])
				
	#arrange output image
	recognized_count = len(recognized_db)
	for x in range(recognized_count):
		xval, yval, orientation = recognized_db[x]['x_position'], recognized_db[x]['y_position'], recognized_db[x]['orientation']
		#rebuild theta based on orientation
		vector1, vector2 = getvectors(xval, yval, ((np.pi/2) - np.radians(orientation)))
		cv2.circle(output_image, (xval, yval), 20, (255,255,255))
		cv2.line(output_image, vector1, vector2, (255,255,255))
	return output_image
