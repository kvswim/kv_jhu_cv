#Kyle Verdeyen
#kverdey1@jhu.edu
#Computer Vision EN.601.461
#Assignment 1
#Programming section 2, test_line_finder
#test_line_finder: test module for p5, p6, p7
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from p5 import p5
from p6 import p6
from p7 import p7

hough_simple_1 = cv2.imread('hough_simple_1.pgm', 0)
hough_simple_2 = cv2.imread('hough_simple_2.pgm', 0)
hough_complex_1 = cv2.imread('hough_complex_1.pgm', 0)

hough_simple_1_edge_image = p5(hough_simple_1)
hough_simple_1_edge_thresh_image, hough_simple_1_hough_image = p6(hough_simple_1_edge_image, 20)
hough_simple_1_line_image = p7(hough_simple_1, hough_simple_1_hough_image, 127)

hough_simple_2_edge_image = p5(hough_simple_2)
hough_simple_2_edge_thresh_image, hough_simple_2_hough_image = p6(hough_simple_2_edge_image, 20)
hough_simple_2_line_image = p7(hough_simple_2, hough_simple_2_hough_image, 127)

hough_complex_1_edge_image = p5(hough_complex_1)
hough_complex_1_edge_thresh_image, hough_complex_1_hough_image = p6(hough_complex_1_edge_image, 20)
hough_complex_1_line_image = p7(hough_complex_1, hough_complex_1_hough_image, 127)

cv2.imshow('hough_simple_1',hough_simple_1)
cv2.imshow('hough_simple_1_edge_image',hough_simple_1_edge_image)
cv2.imshow('hough_simple_1_edge_thresh_image',hough_simple_1_edge_thresh_image)
cv2.imshow('hough_simple_1_line_image',hough_simple_1_line_image)

cv2.imshow('hough_simple_2', hough_simple_2)
cv2.imshow('hough_simple_2_edge_image', hough_simple_2_edge_image)
cv2.imshow('hough_simple_2_edge_thresh_image', hough_simple_2_edge_thresh_image)
cv2.imshow('hough_simple_2_line_image', hough_simple_2_line_image)

cv2.imshow('hough_complex_1', hough_complex_1)
cv2.imshow('hough_complex_1_edge_image', hough_complex_1_edge_image)
cv2.imshow('hough_complex_1_edge_thresh_image', hough_complex_1_edge_thresh_image)
cv2.imshow('hough_complex_1_line_image', hough_complex_1_line_image)

plt.imshow(hough_simple_1_hough_image)
plt.show()
plt.savefig('hough_simple_1_hough_image.png')
plt.imshow(hough_simple_2_hough_image)
plt.show()
plt.savefig('hough_simple_2_hough_image.png')
plt.imshow(hough_complex_1_hough_image)
plt.show()
plt.savefig('hough_complex_1_hough_image.png')

#plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('hough_simple_1.png',hough_simple_1)
cv2.imwrite('hough_simple_1_edge_image.png',hough_simple_1_edge_image)
cv2.imwrite('hough_simple_1_edge_thresh_image.png',hough_simple_1_edge_thresh_image)
cv2.imwrite('hough_simple_1_line_image.png',hough_simple_1_line_image)

cv2.imwrite('hough_simple_2.png', hough_simple_2)
cv2.imwrite('hough_simple_2_edge_image.png', hough_simple_2_edge_image)
cv2.imwrite('hough_simple_2_edge_thresh_image.png', hough_simple_2_edge_thresh_image)
cv2.imwrite('hough_simple_2_line_image.png', hough_simple_2_line_image)

cv2.imwrite('hough_complex_1.png', hough_complex_1)
cv2.imwrite('hough_complex_1_edge_image.png', hough_complex_1_edge_image)
cv2.imwrite('hough_complex_1_edge_thresh_image.png', hough_complex_1_edge_thresh_image)
cv2.imwrite('hough_complex_1_line_image.png', hough_complex_1_line_image)