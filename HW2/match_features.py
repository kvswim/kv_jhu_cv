import cv2
import numpy as np
import matplotlib.pyplot as plt

def match_features(feature_coords1,feature_coords2,image1,image2):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        feature_coords1 (list of tuples): list of (row,col) tuple feature coordinates from image1
        feature_coords2 (list of tuples): list of (row,col) tuple feature coordinates from image2
        image1 (numpy.ndarray): The input image corresponding to features_coords1
        image2 (numpy.ndarray): The input image corresponding to features_coords2
    Returns:
        matches (list of tuples): list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 0-th feature
                                  in feature_coords2 are determined to be matches, the list should contain (4,0).
    """
    matches = list()
    ncc_descriptor1 = ncc_descriptor(image1, feature_coords1) 
    ncc_descriptor2 = ncc_descriptor(image2, feature_coords2)
    num_descriptors1 = np.shape(ncc_descriptor1)[2]
    num_descriptors2 = np.shape(ncc_descriptor2)[2]
    print(num_descriptors1)
    print(num_descriptors2)
    return matches

def ncc_descriptor(image, feature_coordinates):
    feature_coordinates = np.array(feature_coordinates)
    feature_coordinates = feature_coordinates.T
    rows = feature_coordinates[0]
    columns = feature_coordinates[1]
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = np.shape(grayscale)
    numfeatures = len(rows)
    window_size = 13
    delta = (window_size - 1) / 2
    descriptors = np.zeros((window_size, window_size, numfeatures))
    for x in range(0, numfeatures):
        if rows[x] > delta and columns[x] > delta and columns[x] + delta < width and rows[x] + delta < height:
            descriptor = grayscale[rows[x] - delta - 1 : rows[x] + delta].T[columns[x]- delta - 1 : columns[x]+delta]
            descriptors[:, :, x] = descriptor

    return descriptors
#def compute_ncc(ncc_descriptor1, ncc_descriptor2, descriptor1_index, descriptor2_index):
