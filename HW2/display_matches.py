import numpy as np
import cv2
def display_matches(image1, image2, features1, features2, matches):
	rows1, columns1 = [item[0] for item in features1], [item[1] for item in features1]
	rows2, columns2 = [item[0] for item in features2], [item[1] for item in features2]
	offset = image1.shape[0]
	try:
		output_image = np.concatenate((image1, image2), axis = 1)
	except:
		#output_image = np.concatenate((image1, image2))
		output_image = np.append(image1[:, np.newaxis], image2[:, np.newaxis], axis=1)
	for x in range(0, len(matches)):
		x1 = columns1[matches[x][0]]
		y1 = rows1[matches[x][0]]
		x2 = columns2[matches[x][1]] + offset
		y2 = rows2[matches[x][1]]
		cv2.line(output_image, (x1, y1), (x2, y2), (255, 255, 255))
	return output_image


def display_ransac(image1, image2, features1, features2, matches, affine):
	rows1, columns1 = [item[0] for item in features1], [item[1] for item in features1]
	rows2, columns2 = [item[0] for item in features2], [item[1] for item in features2]
	offset = image1.shape[0]
	output_image = np.concatenate((image1, image2), axis = 1)
	for x in range(0, len(matches)):
		x1 = columns1[matches[x][0]]
		y1 = rows1[matches[x][0]]
		x2 = columns2[matches[x][1]] + offset
		y2 = rows2[matches[x][1]]

		x_coordinates = np.dot(affine, np.array([x1, y1, 1]))
		xdiff = x_coordinates[0] - columns2[matches[x][1]]
		ydiff = x_coordinates[1] - rows2[matches[x][1]]
		if np.absolute(xdiff) < 50 and np.absolute(ydiff) < 50:
			cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0))
		else:
			cv2.line(output_image, (x1, y1,), (x2, y2), (0, 0, 255))
	return output_image