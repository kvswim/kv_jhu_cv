import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_affine_xform(matches,features1,features2,image1,image2):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        matches (list of tuples): list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 0-th feature
                                  in feature_coords2 are determined to be matches, the list should contain (4,0).
        features1 (list of tuples) : list of feature coordinates corresponding to image1
        features2 (list of tuples) : list of feature coordinates corresponding to image2
        image1 (numpy.ndarray): The input image corresponding to features_coords1
        image2 (numpy.ndarray): The input image corresponding to features_coords2
    Returns:
        affine_xform (numpy.ndarray): a 3x3 Affine transformation matrix between the two images, computed using the matches.
    """
    
    affine_xform = np.zeros((3,3))
    rows1, columns1 = [item[0] for item in features1], [item[1] for item in features1]
    rows2, columns2 = [item[0] for item in features2], [item[1] for item in features2]
    if len(matches) < 3:
        return affine_xform

    num_iter = 200
    best_form = np.zeros((6,1))
    best_inlier = -1

    for i in range(0, num_iter):
        randmatch = random.sample(matches, 3)
        x1 = columns1[randmatch[0][0]]
        y1 = rows1[randmatch[0][0]]
        x2 = columns1[randmatch[1][0]]
        y2 = rows1[randmatch[1][0]]
        x3 = columns1[randmatch[2][0]]
        y3 = columns1[randmatch[2][0]]

        xp1 = columns2[randmatch[0][1]]
        yp1 = rows2[randmatch[0][1]]
        xp2 = columns2[randmatch[1][1]]
        yp2 = rows2[randmatch[1][1]]
        xp3 = columns2[randmatch[2][1]]
        yp3 = rows2[randmatch[2][1]]

        
        A = np.array([[x1, y1, 1, 0, 0, 0], [0, 0, 0, x1, y1, 1], [x2, y2, 1, 0, 0, 0], [0, 0, 0, x2, y2, 1], [x3, y3, 1, 0, 0, 0], [0, 0, 0, x3, y3, 1]])

        if np.linalg.cond(A) > 1e+15:
            num_iter += 1
            continue
        
        b = np.array([xp1, yp1, xp2, yp2, xp3,yp3])
        
        #solve least squares
        #temp_affine = np.linalg.inv(A.T*A)*A.T*b
        temp_affine = np.linalg.solve(A,b)
        
        num_inliers = 0

        for j in range(0, len(matches)):
            diff_x = (np.dot(temp_affine[0], columns1[matches[j][0]]) + np.dot(temp_affine[1], rows1[matches[j][0]]) + temp_affine[2]) - columns2[matches[j][1]]
            diff_y = (np.dot(temp_affine[3], columns1[matches[j][0]]) + np.dot(temp_affine[4], rows1[matches[j][0]]) + temp_affine[5]) - rows2[matches[j][1]]
            if np.absolute(diff_x) < 3 and np.absolute(diff_y) <3:
                num_inliers += 1
        if num_inliers > best_inlier:
            xform = [[temp_affine[0], temp_affine[1], temp_affine[2]], [temp_affine[3], temp_affine[4], temp_affine[5]], [0, 0, 1]]
            if np.linalg.cond(xform) < 1e+15:
                best_inlier = num_inliers
                best_form = temp_affine
            else: 
                num_iter += 1
    affine_xform = [[best_form[0], best_form[1], best_form[2]], [best_form[3], best_form[4], best_form[5]], [0, 0, 1]]
    return affine_xform
