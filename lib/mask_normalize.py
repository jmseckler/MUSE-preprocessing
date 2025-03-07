import cv2 as cv
import numpy as np
from scipy.signal import convolve2d

def convolve_with_circle(image: np.ndarray, radius: int = 40) -> np.ndarray:
	if len(image.shape) != 2:
		raise ValueError("Input image must be a grayscale 2D array.")
	
	# Create a circular kernel
	diameter = 2 * radius
	kernel = np.zeros((diameter, diameter), dtype=np.float32)
	
	# Generate a circular mask
	y, x = np.ogrid[-radius:radius, -radius:radius]
	mask = x**2 + y**2 <= radius**2
	kernel[mask] = 1.0
	
	# Normalize the kernel so that it sums to 1 (preserving intensity)
	kernel /= kernel.sum()
	
	# Apply convolution using scipy's convolve2d (or cv2.filter2D)
	convolved_image = convolve2d(image, kernel, mode='same', boundary='symm')
	
	return convolved_image



def find_all_fasicles_and_Return_as_single_images(MASK):
	_, mask = cv.threshold(MASK, 127, 255, cv.THRESH_BINARY)
	
	masks = []
	count = 0
	
	heirarchy, contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	
	means = []
	count = 0
	for h in heirarchy:
		if h.shape[0] > 50:
			masks.append(np.zeros(MASK.shape).astype('uint8'))
			means.append(np.mean(h,axis=0)[0].astype('int'))
			for i in range(h.shape[0]):
				masks[count][h[i][0][1]][h[i][0][0]] = 255
				start_pixel = tuple(h[i - 1][0])
				end_pixel = tuple(h[i][0])
				cv.line(masks[count], start_pixel, end_pixel, 255, 1)
			count += 1
	
	n = len(masks)
	h, w = masks[0].shape[:2]
	for i in range(n):
		mask_fill = np.zeros((h+2, w+2), np.uint8)
		pixel = (means[i][0],means[i][1])
		cv.floodFill(masks[i], mask_fill, pixel, 255)
		masks[i] = np.clip(cv.bitwise_not(masks[i]),0,255)
		masks[i] = 255 - masks[i]
		masks[i] = np.clip(masks[i],0,1)
	
	return masks

def get_fname(index):
	if index < 10:
		f = 'mask_0000' + str(index)
	elif index < 100:
		f = 'mask_000' + str(index)
	elif index < 1000:
		f = 'mask_00' + str(index)
	else:
		f = 'mask_0' + str(index)
	return f

def normalize_fasicles_on_convolution(image,maskPath):
	mask = cv.imread(maskPath,cv.IMREAD_GRAYSCALE)
	individual_masks = find_all_fasicles_and_Return_as_single_images(mask)

	cImage = convolve_with_circle(image)

	n = len(individual_masks)
	
	binary = []
	for i in range(n):
		binary.append(individual_masks[i].astype(bool))
	
	allMeans = []
	for i in range(n):
		tmp = image[binary[i]]
		allMeans.append(tmp.mean())
	
	allMeans = np.array(allMeans)
	NewValue = np.amax(allMeans)
	
	new_image = np.zeros(image.shape)
	
	new_image = image - cImage
	new_image = new_image + NewValue
	
	bMask = mask // 255
	
	new_image = new_image * bMask
	
	negativeMask = 255 - mask
	negativeMask = negativeMask // 255
	negativeIMG = image * negativeMask
	new_image = new_image + negativeIMG
	
	return new_image


def normalize_means_of_faciles_to_each_other(image,maskPath):
	mask = cv.imread(maskPath,cv.IMREAD_GRAYSCALE)
	individual_masks = find_all_fasicles_and_Return_as_single_images(mask)
	n = len(individual_masks)
	
	binary = []
	for i in range(n):
		binary.append(individual_masks[i].astype(bool))
	
	allMeans = []
	for i in range(n):
		tmp = image[binary[i]]
		allMeans.append(tmp.mean())
	
	allMeans = np.array(allMeans)
	NewValue = np.amax(allMeans)
	new_image = np.zeros(image.shape)
	for i in range(n):
		tmpImg = image - allMeans[i]
		tmpImg = tmpImg + NewValue
		tmpImg = tmpImg * individual_masks[i]
		new_image = tmpImg + new_image
	negativeMask = 255 - mask
	negativeMask = negativeMask // 255
	negativeIMG = image * negativeMask
	new_image = new_image + negativeIMG
	return new_image

