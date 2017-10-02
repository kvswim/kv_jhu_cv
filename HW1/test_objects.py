#Kyle Verdeyen
#kverdey1@jhu.edu
#Computer Vision EN.601.461
#Assignment 1
#Programming section 1, p1-4
#test_objects.py: test imaging functions p1-p4.
import numpy as np
import matplotlib.pyplot as plt
import cv2
from p1 import p1
from p2 import p2
from p3 import p3
from p4 import p4


two_objects = cv2.imread('two_objects.pgm', 0) #load grayscale image in grayscale mode
many_objects_1 = cv2.imread('many_objects_1.pgm', 0)
many_objects_2 = cv2.imread('many_objects_2.pgm', 0)
two_objects_binary = p1(two_objects, 127) #create binary_in, which is binary_out of p1
two_objects_labels = p2(two_objects_binary)
two_objects_database, two_objects_output_image = p3(two_objects_labels)
mo1_output_image = p4(p2(p1(many_objects_1,127)), two_objects_database)
mo2_output_image = p4(p2(p1(many_objects_2, 127)), two_objects_database)

plt.imshow(two_objects_labels)
cv2.imshow('p1: two_objects_binary', two_objects_binary)
cv2.imshow('p2: labels_out (cv2 image)', two_objects_labels)
cv2.imshow('p3: two_objects_output_image', two_objects_output_image)
cv2.imshow('p4: output_image (recognized objects of MO1)', mo1_output_image)
cv2.imshow('p4: output_image (recognized objects of MO2', mo2_output_image)
plt.show() #needs to be closed manually
cv2.waitKey(0)
cv2.destroyAllWindows() #close all image windows when key pressed

cv2.imwrite('two_objects_binary.png', two_objects_binary)
cv2.imwrite('labels_out (cv2 image).png', two_objects_labels)
cv2.imwrite('two_objects_output_image.png', two_objects_output_image)
cv2.imwrite('output_image (recognized objects of MO1).png', mo1_output_image)
cv2.imwrite('output_image (recognized objects of MO2).png', mo2_output_image)