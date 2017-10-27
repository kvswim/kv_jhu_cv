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
    potential_match = np.array([])

    for x in range(0, num_descriptors1):
        index = -1
        max_dist = -1
        for y in range(0, num_descriptors2):
            ncc = compute_ncc(ncc_descriptor1, ncc_descriptor2, x, y)
            if ncc > max_dist:
                max_dist = ncc
                index = y
        if index > -1:
            match = [x, index]
            potential_match = np.append(potential_match, match)
    for x in range(0, num_descriptors2):
        index = -1
        max_dist = -1    
        for y in range(0, num_descriptors1):
            ncc = compute_ncc(ncc_descriptor1, ncc_descriptor2, y, x)
            if ncc > max_dist:
                max_dist = ncc
                index = y
        if index > -1:
            union = -1
            
            for z in range(0, len(potential_match)):
                if potential_match[z] == x:
                    union = z
            if union > -1:
                match = [index, x]
                matches.append(match)
    return matches

def ncc_descriptor(image, feature_coordinates):
    feature_coordinates = np.array(feature_coordinates)
    feature_coordinates = feature_coordinates.T
    rows = feature_coordinates[:][0]
    columns = feature_coordinates[:][1]
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

def compute_ncc(ncc_descriptor1, ncc_descriptor2, descriptor1_index, descriptor2_index):
    normal1 = compute_normal(ncc_descriptor1, descriptor1_index)
    normal2 = compute_normal(ncc_descriptor2, descriptor2_index)
    ncc = np.multiply(normal1, normal2)
    ncc = np.sum(np.sum(ncc))
    return ncc

def compute_normal(descriptor, index):
    normal = descriptor[:, :, index]
    U, s, V = np.linalg.svd(normal)
    maximum = np.amax(s)
    normal = np.divide(normal, maximum+1e-16)
    return normal