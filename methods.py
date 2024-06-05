import zarr, glob, os, shutil, math, json
import scipy as sp
import skimage as sk
import cv2 as cv
import numpy as np

#Ian Specific Library
import dask.array as da
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscale
import skimage


#input_path = '/media/revadata/data/REVA/'
#output_path = './output/'
#input_path = '/media/revadata/data/REVA/SR005/'
input_path =  '/media/james/T9/data/'

#output_path = '/media/revadata/working/data/'
output_path = '/media/james/T9/process/'


threshhold = 128
img_size = 2500
img_step = 200

def find_unprocessed_data_folders():
	data_list = glob.glob(input_path + "*")
	finished_list = glob.glob(output_path + "*")
	flist = []
	for fname in data_list:
		if fname not in finished_list:
			flist.append(fname.split("/")[-1])
	return flist

def zarr_image_lister(path):
	path = input_path + path + "/M*"
	raw = glob.glob(path)
	
	flist = []
	for fname in raw:
		if fname.endswith(("zarr")):
			flist.append(fname)
	flist = sorted(flist)
	return flist

def get_image_from_zarr(path):
	zfile = zarr.open(path)
	try:
		return zfile["/muse/stitched/"], zfile.attrs
	except KeyError:
		print("Filename, "+path+" is corrupted and did not produce a file from zarr file...")
		return None, None

def replace_directory(directory):
	if os.path.isdir(directory):
		shutil.rmtree(directory)
	os.makedirs(directory)


def find_difference_between_images(img1,img2):
	diff = img1 - img2
	diff = np.absolute(diff)
	n = img1.shape[0] * img1.shape[1]
	diff = np.sum(diff)
	return diff / n

def find_crop_positions(img):
	img = (img/16).astype('uint8')
	
    #Put in step to adjust mean to threshhold, this may cause massive issues, check it stupid.
    #Define threshold  in terms of mean and std of the individual image 

	ret, thresh = cv.threshold(img, threshhold, 255, 0)
	heirarchy, contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	size = -1
	cluster = []
	
	for h in heirarchy:
		if len(h) > size:
			size = len(h)
			cluster = h
	
	x = 0
	y = 0
	
	x_par = [10000000,-1]
	y_par = [10000000,-1]
	
	for pos in cluster:
		x += pos[0][0]
		y += pos[0][1]
		if pos[0][0] < x_par[0]:
			x_par[0] = pos[0][0]
		if pos[0][0] > x_par[1]:
			x_par[1] = pos[0][1]
		
		if pos[0][1] < y_par[0]:
			y_par[0] = pos[0][0]
		if pos[0][1] > y_par[1]:
			y_par[1] = pos[0][1]
			
	if len(cluster) == 0:
		return None

	x = int(x / size)
	y = int(y / size)

	width = x_par[0] - x_par[1]
	height = y_par[0] - y_par[1]
	
	radius = img_size
	while width >= 2 * radius or height >= 2 * radius:
		radius += img_step
	
	return [x, y, radius]

def crop_img(image,x,y,r):
	if 2 * r + 2 > np.amin(image.shape):
		r = int(np.amin(image.shape) / 2 - 2)
	
	x_min = x - r
	x_max = x + r
	y_min = y - r
	y_max = y + r
	
	if x_min < 0:
		x_max -= x_min
		x_min = 0
	if y_min < 0:
		y_max -= y_min
		y_min = 0
	
	if x_max > image.shape[1]:
		diff = x_max - image.shape[1] + 1
		x_max = x_max - diff
		x_min = x_min - diff
	if y_max > image.shape[0]:
		diff = y_max - image.shape[0] + 1
		y_max = y_max - diff
		y_min = y_min - diff
	
	cropped = image[y_min:y_max,x_min:x_max]
	
	if cropped.shape[0] < 2 * r or cropped.shape[1] < 2 * r:
		embed = np.zeros((2*r,2*r))
		offset_x = 2 * r - cropped.shape[0]
		offset_y = 2 * r - cropped.shape[1]
		
		x_min_embedd = int(offset_x / 2)
		x_max_embedd = x_min_embedd + cropped.shape[0]

		y_min_embedd = int(offset_y / 2)
		y_max_embedd = y_min_embedd + cropped.shape[0]
		
		embed[x_min_embedd:x_max_embedd,y_min_embedd:y_max_embedd] = image[y_min:y_max,x_min:x_max]
		cropped = embed
		
	return cropped
	
def segment_out_the_nerve(img):
	img = (img/16).astype('uint8')
	mean = np.mean(img)
	img = img - mean
	img = img + threshhold
	
	std = np.std(img)
	
	img_threshhold = threshhold + 0.25 * std
	
	ret, thresh = cv.threshold(img, img_threshhold, 255, 0)
	thresh = np.array(thresh, np.uint8)
	
	heirarchy, contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	
	nerves_size = -1
	nerve_cluster = -1
	for h in heirarchy:
		if len(h) > nerves_size:
			nerves_size = len(h)
			nerve_cluster = h

	
	dx = [100000,-1]
	dy = [100000,-1]
	
	x = 0
	y = 0
	counter = 0
	
	nerve = np.zeros(img.shape)
	
	for pos in nerve_cluster:
		nerve[pos[0][1]][pos[0][0]] = 255
		if pos[0][0] < dx[0]:
			dx[0] = pos[0][0]
		elif pos[0][0] > dx[1]:
			dx[1] = pos[0][0]
		if pos[0][1] < dy[0]:
			dy[0] = pos[0][1]
		elif pos[0][1] > dx[1]:
			dy[1] = pos[0][1]
		x += pos[0][0]
		y += pos[0][1]
		counter += 1
	
	x = x / counter
	y = y / counter
	
	for i in range(len(nerve_cluster)):
		start_pixel = tuple(nerve_cluster[i - 1][0])
		end_pixel = tuple(nerve_cluster[i][0])
		cv.line(nerve, start_pixel, end_pixel, 255, 1)

#	cv.imwrite(f"./output/raw_{z}.png",img)

#	cv.imwrite(f"./output/nerve_{z}.png",nerve)
	for i in range(nerve.shape[0]):
		if nerve[i][0] > 0:
			nerve[i][1] = 255
			nerve[i][0] = 0
		if nerve[i][-1] > 0:
			nerve[i][-2] = 255
			nerve[i][-1] = 0
	
	nerve = nerve.astype('uint8')
	pixel = (0,0)
	_, mask = cv.threshold(nerve, 1, 255, cv.THRESH_BINARY)
	h, w = nerve.shape[:2]
	mask_fill = np.zeros((h+2, w+2), np.uint8)
	
	cv.floodFill(mask, mask_fill, pixel, 255)
	
	mask = np.clip(cv.bitwise_not(mask),0,1)
	
	diameter = np.amax(np.array(np.abs(dx[1]-dx[0]),np.abs(dy[1]-dy[0])))
	
	return [x,y,diameter], mask



	



def save_image(name,path,img):
	cv.imwrite(path + name + '.png',img)

def save_zarr_file(path,z):
	zarr.save(path + '/data' + '.zarr', z)

def load_zarr_file(path):
	return zarr.load(path + '/data' + '.zarr')

def bleach_and_gamma_initial_variables(img,bitrate = 16):
	pixel = 256 * bitrate - 1
	scale = 16 / bitrate
	
	mean_first_slice = np.mean(img[0]) * scale
	
	try:
		gamma = math.log(0.5*pixel)/math.log(mean_first_slice)
	except ValueError:
		print(pixel,mean_first_slice,np.sum(img))
		quit()
	
	
	
	return mean_first_slice, gamma

def save_arributes(name,data):
	path = name + '/attibutes.json'
	with open(path, 'w') as json_file:
		json.dump(data, json_file)

def load_attributes(name):
	path = name + '/attibutes_' + '.json'
	with open(path, 'r') as json_file:
		data = json.load(json_file)
	return data

def save_sample_png(path,arr):
	path = path + '/png/'
	replace_directory(path)
	n = arr.shape[0]
	
	if n < 100:
		k = 1
	else:
		k = int(n/100)
	
	for i in range(n):
		if i % k == 0 and np.sum(arr[i]) > 0:
			print(np.amax(arr[i] / 16),np.amin(arr[i] / 16))
			save_image(str(i),path,arr[i] / 16)
	

def save_data(name,arr,data):
	path = output_path + name
	replace_directory(path)
	save_arributes(path,data)

def load_data(name):
	path = output_path + name
	arr = load_zarr_file(path)
	data = load_attributes(path)
	return arr,data


def crop_black_border(image):
	# Find the first and last non-black pixels along rows
	row_sum = np.sum(image, axis=1)
	first_row = np.argmax(row_sum > 0)
	last_row = len(row_sum) - np.argmax(row_sum[::-1] > 0)
	
	# Find the first and last non-black pixels along columns
	col_sum = np.sum(image, axis=0)
	first_col = np.argmax(col_sum > 0)
	last_col = len(col_sum) - np.argmax(col_sum[::-1] > 0)
	
	# Crop the image to the non-black region
	cropped_image = image[first_row:last_row, first_col:last_col]
	return cropped_image

def crop_down_to_size(image,size):
	radius = int(size / 2)
	
	x0 = int(image.shape[0] / 2)
	y0 = int(image.shape[1] / 2)
	
	x1 = x0 - radius
	x2 = x0 + radius
	y1 = y0 - radius
	y2 = y0 + radius
	
	cropped_image = image[x1:x2,y1:y2]
	return cropped_image

def add_smaller_image_to_larger(smaller_image,size):
	# Get the shape of the larger and smaller images
	larger_shape = (size,size)
	larger_image = np.zeros(larger_shape)
	smaller_shape = smaller_image.shape
	
	# Calculate the position to place the smaller image in the center of the larger image
	x_start = (larger_shape[0] - smaller_shape[0]) // 2
	y_start = (larger_shape[1] - smaller_shape[1]) // 2
	x_end = x_start + smaller_shape[0]
	y_end = y_start + smaller_shape[1]
	    
	# Create a copy of the larger image to avoid modifying the original image
	result_image = np.copy(larger_image)
	
	# Add the smaller image to the center of the larger image
	result_image[x_start:x_end, y_start:y_end] += smaller_image
	return result_image


def coregister(img1,img2):
	shift, err, diff_phase = sk.registration.phase_cross_correlation(img1,img2)	
	img2 = sp.ndimage.shift(img2,shift)
	return img2, shift

def center_on_nerve(img,y,x):
	x0 = int(x - (img.shape[0] / 2))
	y0 = int(y - (img.shape[1] / 2))
	
	shift = (x0,y0)
	img = sp.ndimage.shift(img,shift)
	return img

def process_image(img,mean,radius,align_image=None):
	image = img - np.mean(img)
	img = img + mean

	size = np.amax(img.shape)
	
	img = add_smaller_image_to_larger(img,size)
	
	crop,mask = segment_out_the_nerve(img)
	
	x0 = int(crop[0])
	y0 = int(crop[1])
	
	image = center_on_nerve(img,x0,y0)

	if radius > size:
		image = add_smaller_image_to_larger(image,radius)
	else:
		image = crop_down_to_size(image,radius)

#	image = crop_down_to_size(image,radius)
	
	if align_image is None:
		print("Original,",x0,y0)
	else:
		image, s = coregister(align_image,image)
	return image



#Put Ian Code Here

def build_coordinate_transforms_metadata_transformation_from_pixel():
	ct1 = [{"type": "scale", "scale": [12, 9, 9]}]
	ct5 = [{"type": "scale", "scale": [12, 45, 45]}]
	ct10 = [{"type": "scale", "scale": [12, 90, 90]}]
	
	ct = [ct1, ct5, ct10]
	
	#build multiscales metadata
	axes = [
		{"name": "z", "type": "space", "unit": "micrometer"},
		{"name": "y", "type": "space", "unit": "micrometer"},
		{"name": "x", "type": "space", "unit": "micrometer"}]
	
	datasets = [
		{"coordinateTransformations": ct1, "path": "0"},
		{"coordinateTransformations": ct5, "path": "1"},
		{"coordinateTransformations": ct10, "path": "2"}]
	
	multiscales = [{
		"name": "/",
		"version": "0.4",
		"axes": axes,
		"datasets": datasets,
	}]
	
	return multiscales, ct






	
	
	
