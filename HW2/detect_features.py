import cv2
import numpy as np
import matplotlib.pyplot as plt
from nonmaxsuppts import nonmaxsuppts

def detect_features(image):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        image (numpy.ndarray): The input image to detect features on. Note: this is NOT the image name or image path.
    Returns:
        pixel_coords (list of tuples): A list of (row,col) tuples of detected feature locations in the image
    """
    pixel_coords = list()
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gaussian_kernel = cv2.getGaussianKernel(3, 1)
    sobelx =[[-1, 0 ,1], [-2, 0, 2], [-1, 0, 1]]
    sobelx = np.array(sobelx)
    sobely = sobelx.T
    ix = cv2.filter2D(grayscale, -1, sobelx)
    iy = cv2.filter2D(grayscale, -1, sobely)
    fxx = cv2.filter2D(np.multiply(ix,ix), -1, gaussian_kernel)
    fxy = cv2.filter2D(np.multiply(ix, iy), -1, gaussian_kernel)
    fyy = cv2.filter2D(np.multiply(iy, iy), -1, gaussian_kernel)
    cs = (np.multiply(fxx, fyy) - np.power(fxy, 2)) / (fxx + fyy + (1*10**-16))
    pixel_coords = nonmaxsuppts(cs, 2, 6000)

    return pixel_coords
