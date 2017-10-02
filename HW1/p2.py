#Kyle Verdeyen
#kverdey1@jhu.edu
#Computer Vision EN.601.461
#Assignment 1
#Programming section 1, p2
#p2.py: runs sequential labeling algorithm that segments binary image into connected regions
#4-connected labeling algorithm
import cv2
import numpy as np

#Modeled after StackOverflow disjoint set data structure
#https://stackoverflow.com/questions/3067529/a-set-union-find-algorithm
#represents equivalence table, but unions automatically when items added,
#instead of having to track two different sets. 
class EquivalenceTable(object): 

	def __init__(self):
			self.labels = {} 
			self.objects = {}

	def insert(self, x, y): #definition for adding new entry with union-find done automatically
		labelsx = self.labels.get(x) #find value 1
		labelsy = self.labels.get(y) #find value 2

		if labelsx is not None: #if label1 not empty (isn't in the set)
			if labelsy is not None: #if label2 not empty
				if labelsx == labelsy:#if equivalent
					return #we don't need to do anything

				objectsx = self.objects[labelsx] 
				objectsy = self.objects[labelsy]

				if len(objectsx) < len(objectsy): #set 1 is shorter than set 2
				 	x, labelsx, objectsx, y, labelsy, objectsy = y, labelsy, objectsy, x, labelsx, objectsx
				 	#don't want to do the temp value shuffle
				 	#x = y
				 	#labelsx = labelsy
				 	#objectsx = objectsy
				 	#y = x
				 	#labelsy = labelsx 
				 	#objectsy = objectsx
				#multiline fails to label continuous objects with all same label
				#unions = objectsx | objectsy
				#objectsx = unions
				objectsx |= objectsy #bitwise or assignment, only true if y is in x
				del self.objects[labelsy] #delete the entry for 2
				for z in objectsy: #union 
					self.labels[z] = labelsx #every label position in ob2 gets label1 

			else: #label2 is empty
				self.objects[labelsx].add(y) #add 2 to objects
				self.labels[y] = labelsx #label position y gets label2
		else: # x is not found
			if labelsy is not None: #in y but not x 
				self.objects[labelsy].add(x) #add to equivalence
				self.labels[x] = labelsy
			else: #it's a brand new object
				self.labels[x] = self.labels[y] = x #new label entry
				self.objects[x] = set([x, y]) # add objects to table


#directions: left, up, upleft, e.g. we're first checking bottom right in check4
#this could probably be split out into separate methods, implementation is pretty slow and has a deep cyclical loop
#essentially brute force but whatever works
def p2(binary_in): #return labels_out
	#ret, labels_out = cv2.connectedComponents(binary_in)
	dimensions = np.shape(binary_in)
	rows = dimensions[0]
	columns = dimensions[1]
	labels_out = np.zeros((rows,columns))
	#labels_out = np.empty(dimensions, dtype = int)
	value = 1
	equivalence_table = EquivalenceTable() #see below
	#for location, value in ndenumerate(self.binary_in): can't use this since we need directionality
	for x in range(rows):
		for y in range(columns):
			if binary_in[x][y] > 0: #input value nonzero
				if x == 0: #if we're on the top row
					if y > 0: #not on the leftmost column
						left = labels_out[x][y-1] #check left
						if left > 0: #nonzero left
							labels_out[x][y] = left #assign label
							#value += 1 #increment label
						else: #zero left
							labels_out[x][y] = value
							value += 1
					else: #on the leftmost column
						labels_out[x][y] = value
						value += 1
				elif y == 0: #not on the top row, check if left column
					if x > 0: #not top row
						up = labels_out[x-1][y]
						if up > 0: #up cell is not null
							labels_out[x][y] = up #gets same label as up
						else: #up cell has no value [unlabeled]
							labels_out[x][y] = value #mark working cell
							value += 1 #and increment
					else: #on top row
						labels_out[x][y] = value
						value += 1
				else: #not on the top row, not on the left column
					up = labels_out[x-1][y] #get neighboring pixels since we can't go out of bounds
					upleft = labels_out[x-1][y-1]
					left = labels_out[x][y-1]
					#start check, nothing around us?
					if up == 0 and upleft == 0 and left == 0:
						labels_out[x][y] = value
						value += 1
					#ok, there's stuff around. check if we're bordering both
					elif left > 0 and up > 0 and upleft == 0:
						labels_out[x][y] = up
						if left != up: #check if we need to add to equiv table
							equivalence_table.insert(left, up)
						#don't care otherwise
					#neighboring pixels not both occupied. check cell up
					elif up > 0 and left == 0 and upleft == 0:
						labels_out[x][y] = up
						#neightboring cells unoccupied so don't need to increment labels or add to table
					#not bordering, not top. check cell left
					elif left > 0 and up == 0 and upleft == 0:
						labels_out[x][y] = left
					#left and right empty, check upper left
					elif upleft > 0:
						labels_out[x][y] = upleft
					#no close cells occupied
					else:
						labels_out[x][y] = value
						value += 1
			else: #we're looking at a zero value
				labels_out[x][y] = 0 #don't care, nothing important

	#done with primary first pass, 2nd pass to resolve equivalence
	for x in range(rows):
		for y in range(columns):
			if labels_out[x][y] > 0: #analyze pixels. only care if nonzero, ignore otherwise
				count = 0 #start 
				for z in equivalence_table.objects: #load line
					if labels_out[x][y] in equivalence_table.objects[z]: #if pixel under scrutinization is in this line of the equivalence table
						#generates rudimentary grayscale value based on label
						labels_out[x][y] = int(len(equivalence_table.objects)*(count+1))
						break #loopback
					count += 1 #increment

	return labels_out
