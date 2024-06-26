#import methods as ms
import cv2 as cv
import numpy as np
import sys
inpath = "./input/"
fname = sys.argv[1]

img = cv.imread(inpath + fname + '.png',cv.IMREAD_GRAYSCALE)

def on_one_side_of_mean(array,above = True):
	# Calculate the mean of the array
	mean_value = np.mean(array)
	# Filter values greater than the mean
	if above:
		values_mean = array[array > mean_value]
	else:
		values_mean = array[array < mean_value]
	    
	# Check if there are any values greater than the mean
	if values_mean.size == 0:
		return None  # Return None if no values are greater than the mean
	
	# Calculate the average of the filtered values
	average = np.mean(values_mean)
	
	return average

y_min = int(sys.argv[2])
y_max = int(sys.argv[4])
x_min = int(sys.argv[3])
x_max = int(sys.argv[5])

noise_y_min = int(sys.argv[6])
noise_y_max = int(sys.argv[8])
noise_x_min = int(sys.argv[7])
noise_x_max = int(sys.argv[9])


light = on_one_side_of_mean(img[x_min:x_max, y_min: y_max])
dark = on_one_side_of_mean(img[x_min:x_max, y_min: y_max],False)

noise = np.std(img[noise_x_min:noise_x_max, noise_y_min:noise_y_max])

print("Contrast to Noise Ratio: ",np.abs(light - dark) / noise)
