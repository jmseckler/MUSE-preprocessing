import warnings
warnings.filterwarnings("ignore")

import zarr, glob, os, shutil, math, json, re
import scipy as sp
import skimage as sk
import cv2 as cv
import numpy as np
import tifffile as tiffio

#Ian Specific Library
import dask.array as da
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscale


#input_path = '/media/revadata/data/REVA/'
#output_path = './output/'
#input_path = '/media/revadata/data/REVA/SR005/'
input_path =  '/media/james/T9/data/'

#output_path = '/media/revadata/working/data/'
output_path = '/media/james/T9/process/'

zarr_attr_path = "./.zattrs"
logFileName = 'muse_application.log'

threshhold = 128
img_size = 2500
img_step = 200
sample = 50

bytes_per_image = 24019288


def find_unprocessed_data_folders(path=""):
	if path == "":
		inpath = input_path
	else:
		inpath = path
	data_list = glob.glob(inpath + "*")
	finished_list = glob.glob(output_path + "*")
	flist = []
	for fname in data_list:
		if fname not in finished_list:
			flist.append(fname.split("/")[-1])
	return flist

def zarr_image_lister(path,inpath = ""):
	if inpath == "":
		path = input_path + path + "/M*"
	else:
		path = inpath + path + "/M*"
	raw = glob.glob(path)
	
	flist = []
	for fname in raw:
		if fname.endswith(("zarr")):
			flist.append(fname)
	flist = sorted(flist)
	return flist




def get_image_from_zarr(path):
	try:
		zimg = da.from_zarr(path, component="muse/stitched/")
		return zimg, None
	except:
		if save_single_panel_tiff_as_zarr_file(zpath):
			zimg = da.from_zarr(path, component="muse/stitched/")
			return zimg, None
		else:
			print("Filename, "+path+" is corrupted or incorrect and did not produce a file from zarr file...")
			return None, None
		

def get_just_images_from_zarr(path):
	img, zattr = get_image_from_zarr(path)
	return img


def make_directory(directory):
	if not os.path.isdir(directory):
		os.makedirs(directory)

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

def segment_out_the_nerve_png(img):
	mean = np.mean(img)
	img = img - mean
	img = img + threshhold
	
	std = np.std(img)
	
	img_threshhold = threshhold + 0.00 * std
	
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
	
	return mask




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
	path = name + '/attibutes.json'
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

def add_scalebar_to_image(image,scale):
	scalebar = cv.imread("./img/scalebar.png",cv.IMREAD_GRAYSCALE)
	scalebar = scalebar / 255
	scalebar = scalebar * scale
	
	
	# Calculate the position to place the smaller image in the center of the larger image
	x_start = image.shape[0] - scalebar.shape[0] - 100
	y_start = image.shape[1] - scalebar.shape[1] - 100
	x_end = image.shape[0] - 100
	y_end = image.shape[1] - 100
	
	bar = np.zeros(image.shape)
	bar[x_start:x_end, y_start:y_end] += scalebar	
	mask = bar > 0
	
	# Create a copy of the larger image to avoid modifying the original image
	result_image = np.copy(image)
	
	# Add the smaller image to the center of the larger image
	result_image[mask] = bar[mask]
	return result_image


def coregister(img1,img2):
	shift, err, diff_phase = sk.registration.phase_cross_correlation(img1,img2)	
	img2 = sp.ndimage.shift(img2,shift)
	return img2, shift

def shiftImage(img,shift):
	img = sp.ndimage.shift(img,shift)
	return img

def center_on_nerve(img,y,x):
	x0 = int(x - (img.shape[0] / 2))
	y0 = int(y - (img.shape[1] / 2))
	
	shift = (x0,y0)
	img = sp.ndimage.shift(img,shift)
	return img

def process_image(img,mean,radius,align_image=None):
	img = img - np.mean(img)
	img = img + mean
	
	size = np.amax(img.shape)
	
	img = add_smaller_image_to_larger(img,size)
	
	crop,mask = segment_out_the_nerve(img)
	
#	mask = mask * 255
	
#	cv.imwrite(f"./output/mask.png",mask)
	
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


def calculate_mean_intensity(filelist):
	counter = 0
	means = []
	for z in filelist:
		try:
			means, counter = load_image_and_get_mean_as_array(z,counter,means)
		except TypeError:#This is where we need to go in and make it revert to tiff stack
			pass
	
	means = np.array(means)
	m = np.average(means)
	std = np.std(means)
	return m, std

def load_image_and_get_mean_as_array(z,counter,means):
	img, attrs = get_image_from_zarr(z)
	
	n = len(img)
	for i in range(n):
		if counter % sample == 0:
			mtemp = np.mean(img[i])
			if mtemp > 0:
				means.append(mtemp)
		counter += 1
	return means,counter

def normalize_to_mean(img,mean):
	image = img - np.mean(img)
	image = image + mean
	image = np.clip(image,0,4095)
	return image

def normalize_mean_and_enhance_contrast(img,mean,factor):
#	print(np.mean(img),mean)
	image = img - np.mean(img)
	image = factor * image	
	image = image + mean
	
	# New intensity = contrast_factor * (Old intensity - 127) + 127
	
	image = np.clip(image,0,4095)
	return image
	
def copy(varr):
	return varr

#Put Ian Code Here

def build_coordinate_transforms_metadata_transformation_from_pixel():
	ct1 = [{"type": "scale", "scale": [12, 0.9, 0.9]}]
	ct5 = [{"type": "scale", "scale": [12, 4.5, 4.5]}]
	ct10 = [{"type": "scale", "scale": [12, 9, 9]}]
	
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

def create_basic_zarr_file(path,fname):
	zarr_path = path + '/' + fname + '.zarr'
	if os.path.isdir(zarr_path):
		shutil.rmtree(zarr_path)
	store = zarr.DirectoryStore(zarr_path, dimension_separator='/')
	root = zarr.group(store=store, overwrite=True)
	data = root.create_group('data')
	return data
	

def remove_directory(directory):
	if os.path.isdir(directory):
		shutil.rmtree(directory)

def copy_directory(src,dst):
	shutil.copytree(src,dst)

def copy_zarr_attr(path,zname):
	dst = path + '/' + zname + '.zarr/.zattrs'
	shutil.copyfile(zarr_attr_path, dst)

def shape_definer(n,x,y,scale):
	zshape = (n,int(x / scale),int(y / scale))
	zchunk = (4,int(x / scale),int(y / scale))
	return zshape, zchunk


def normalize_image_by_column(img,pos):
	subset = np.array(img[pos[0]:pos[1]])
	column_mean_values = np.mean(subset, axis=0)
	column_means = np.tile(column_mean_values, (img.shape[0], 1))
	total_mean = np.mean(subset)
	img = img - column_means
	img = img + total_mean
	img = np.clip(img,0,4095)
	return img


def findAllZarrs(path):
	flist = glob.glob(path + "*.zarr")
	
	allRuns = []
	for fname in flist:
		run = fname.split('.')[0]
		run = run.split('_')[-1]
		allRuns.append(run)
	return allRuns

def findAllDir(path):
	flist = glob.glob(path + "*")
	
	allRuns = []
	for fname in flist:
		if os.path.isdir(fname):
			run = fname.split('/')[-1]
			allRuns.append(run)
	return allRuns



def image_histogram(image, bitdepth = 4096):
	image_array = np.array(image)
	flattened_array = image_array.flatten()
	histogram, bin_edges = np.histogram(flattened_array, bins=bitdepth, range=(0, bitdepth))
	
	return histogram

def logFileLoader(zarrPath):
	path = zarrPath + '/' + logFileName
	
	
	panels = {} #Form of Run#: [hPanels, vPanels]
	XYPositions = {} #Form of Run#:[[X1,Y1],[X2,Y2],...]
	ZPositions = {} #Form of Run#:[Z1,Z2,...]
	voxelSize = {} #Form of Run#: (x,y,z) Voxels
	imageSize = {} #form of Run#:[rows,col]
	exposureTime = {} #form of Run#:Time
	runLength = {} #Form of Run#:[Run Length, Final Slice Made, Images Expected, Images Taken]
	trimLength = {} #Form of Trim#:[Run Length,Date Started, Dated Ended]
	dates = {} #Form of Run#:[Start Date/Time, End Date/Time]
	history = [] #Form of ('run',Run#) or ('trim',Trim#)
	
	#Opens the Raw text file form 
	rawFile = open(path, 'r')
	#regular expressions for capturing information from various row types
	cycletype = re.compile(r"(?P<type>[A-Z]+) CYCLE")
	cyclenum = re.compile(r"CYCLE (?P<cycle>\d+)")
	posre = re.compile(r"\d+\.\d+")
	rowcol = re.compile(r"\(rows, cols\)\: (?P<rows>\d+) (?P<cols>\d+)")
	skipslices = re.compile(r"Skipping imaging every (?P<slices>\d+)")
	imgexpected = re.compile(r"This will generate (?P<imgs>\d+)")
	acqstopped = re.compile(r"Acquisition cycle stopped after (?P<slices>\d+)")
	dateandtime = re.compile(r"\d\d/\d\d/\d\d\d\d \d\d:\d\d:\d\d [AP]M")
	trimstop = re.compile(r"Stopped trimming after (?P<slices>\d+)")
	trimcomp = re.compile(r"Completed trimming for (?P<slices>\d+)")
	exposure = re.compile(r"Set eposure time to (?P<time>\d+)")
	#initialize variables used to track data about runs and trims
	expectedImgs = None
	skipImg = None
	trimNum = 0
	currentRun = None
	sizeXY = None
	runstart = None
	trimstart = None
	runslices = None
	finalSlice = None
	imagesTaken = None
	#Reads in the file and rips the while thing into a list, ready for parsing
	for row in rawFile:
		if cycletype.search(row):
			t = dateandtime.search(row)
			m = cycletype.search(row)
			m2 = cyclenum.search(row)
			if m['type'] == 'IMAGING':
				#if current run is not None, a new run is starting but the previous run has not yet ended so we end it
				#when this happens we are not able to determine how many images were actually taken
				if currentRun != None:
					runLength[currentRun] = [runslices, runslices, expectedImgs, imagesTaken]
					dates[currentRun] = [runstart, t[0]]
					runslices = None
					finalSlice = None
					expectedImgs = None
					imagesTaken = None
					runstart = None
					currentRun = None
				currentRun = int(m2['cycle'])
				runstart = t[0]
				currentRun = f"{runstart} run {currentRun}"
				history.append(('run', currentRun))
			else:
				trimNum += 1
				trimstart = t[0]
				history.append(('trim', trimNum))
		elif 'XY positions are' in row:
			if currentRun == None:
				print("An error has ocurred, looking for XY positions but run is None")
			xs = {}
			ys = {}
			xys = []
			m = posre.findall(row)
			hpanels = 0
			vpanels = 0
			for i in range(0, len(m), 2):
				x = float(m[i])
				y = float(m[i+1])
				if x in xs:
					xs[x] += 1
				else:
					xs[x] = 1
				if y in ys:
					ys[y] += 1
				else:
					ys[y] = 1
				xys.append([x,y])
			XYPositions[currentRun] = xys
			hpanels = max(ys.values())
			vpanels = max(xs.values())
			panels[currentRun] = [hpanels, vpanels]
		elif 'Z positions are' in  row:
			if currentRun == None:
				print("An error has ocurred, looking for Z positions but run is None")
			ZPositions[currentRun] = [float(x) for x in posre.findall(row)]
		elif 'Pixel size' in row:
			if currentRun == None:
				print("An error has ocurred, looking for XY pixel size but run is None")
			sizeXY = float(posre.findall(row)[0])
			m = rowcol.search(row)
			totalRows = int(int(m['rows']) * (0.8 * panels[currentRun][0] - 0.2 ))
			totalCols = int(int(m['cols']) * (0.8 * panels[currentRun][1] - 0.2 ))
			imageSize[currentRun] = [totalRows, totalCols]
		elif "Skipping imaging every" in row:
			if currentRun == None:
				print("An error has ocurred, looking for skipped slices but run is None")
			if sizeXY == None:
				print("An error has ocurred, looking for skipped slices but xy pixel size is None")
			m = skipslices.search(row)
			skipImg = int(m['slices'])
			sizeZ = 3 * (skipImg + 1)
			voxelSize[currentRun] = (sizeXY, sizeXY, sizeZ)
		elif "This will generate" in row:
			m = imgexpected.search(row)
			expectedImgs = int(m['imgs'])
			if skipImg == None:
				print("An error has ocurred, looking for generated images but skipped slices is None")
			runslices = expectedImgs * (skipImg + 1)
			finalSlice = runslices
		elif "Acquisition cycle stopped after" in row:
			m = acqstopped.search(row)
			t = dateandtime.search(row)
			finalSlice = int(m['slices'])
			imagesTaken = int(runslices/(skipImg + 1))
			if currentRun == None:
				print("An error has ocurred, looking for acquisition stopped but current run is None")
			dates[currentRun] = [runstart, t[0]]
			runLength[currentRun] = [runslices, finalSlice, expectedImgs, imagesTaken]
			runslices = None
			finalSlice = None
			expectedImgs = None
			imagesTaken = None
			runstart = None
			currentRun = None
		elif "Completed acquisition cycle" in row:
			t = dateandtime.search(row)
			if currentRun == None:
				print("An error has ocurred, looking for acquisition completed but current run is None")
			dates[currentRun] = [runstart, t[0]]
			imagesTaken = int(runslices/(skipImg + 1))
			runLength[currentRun] = [runslices, runslices, expectedImgs, imagesTaken]
			runslices = None
			finalSlice = None
			expectedImgs = None
			imagesTaken = None
			runstart = None
			currentRun = None
		elif "Stopped trimming after" in row:
			m = trimstop.search(row)
			t = dateandtime.search(row)
			if trimstart == None:
				print("An error has ocurred, looking for trim length but trim start is None")
			trimLength[trimNum] = [int(m['slices']), trimstart, t[0]]
			trimstart = None
		elif "Completed trimming for" in row:
			m = trimcomp.search(row)
			t = dateandtime.search(row)
			if trimstart == None:
				print("An error has ocurred, looking for trim length but trim start is None")
			trimLength[trimNum] = [int(m['slices']), trimstart, t[0]]
			trimstart = None
		elif "Set eposure time" in row:
			m = exposure.search(row)
			if currentRun == None:
				print("An error has ocurred, looking for acquisition completed but current run is None")
			exposureTime[currentRun] = int(m['time'])
		else:
			continue
	
	masterFile = {'runs':{},'trims':{},'names':{},'panelNumbers':{},'history':[],'runList':[]}
	
	#Compiles all data for the Runs
	for run in panels:
		masterFile['runList'].append(run)
		masterFile['names'][run] = run.split(' ')[-1]
		masterFile['panelNumbers'][run] = panels[run][0] * panels[run][1]
		
		masterFile['runs'][run] = {}
		masterFile['runs'][run]['panels'] = panels[run]
		
		try:
			masterFile['runs'][run]['voxel'] = voxelSize[run]
		except KeyError:
			print(f"KeyError in run {run} for voxelSize")

		try:
			masterFile['runs'][run]['size'] = imageSize[run]
		except KeyError:
			print(f"KeyError in run {run} for ImageSize")


		try:
			masterFile['runs'][run]['exposure'] = exposureTime[run]
		except KeyError:
			print(f"KeyError in run {run} for Exposure Time")

		try:
			masterFile['runs'][run]['length'] = {'total cuts':runLength[run][1],'expected cuts':runLength[run][0],'total images':runLength[run][3],'expected images':runLength[run][2]}
		except KeyError:
			print(f"KeyError in run {run} for Run Length")


		try:
			masterFile['runs'][run]['start'] = dates[run][0]
		except KeyError:
			print(f"KeyError in run {run} for Start Time")

		try:
			masterFile['runs'][run]['end'] = dates[run][1]
		except KeyError:
			print(f"KeyError in run {run} for End Time")
		
		try:
			masterFile['runs'][run]['XY_positions'] = XYPositions[run]
		except KeyError:
			print(f"KeyError in run {run} for XYPositions")
		
		try:
			masterFile['runs'][run]['Z_positions'] = ZPositions[run]
		except KeyError:
			print(f"KeyError in run {run} for ZPositions")
	
	#Compiles all data for the Trimming Cycles
	for run in trimLength:
		masterFile['trims'][run] = {}
		masterFile['trims'][run]['length'] = trimLength[run][0]
		masterFile['trims'][run]['start'] = trimLength[run][1]
		masterFile['trims'][run]['end'] = trimLength[run][2]
	
	masterFile['history'] = history
	return masterFile

def save_single_panel_tiff_as_zarr_file(zpath):
#	zpath = inPath + 'MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
	zarrNumber = zpath.split('_')[-1]
	zarrNumber = zarrNumber[:-5]
	listPath = zpath.split('/')
	listPath.pop()
	
	inPath = ''
	for l in listPath:
		inPath += l + '/'
	
	tiffPath = inPath + 'MUSE_acq_' + zarrNumber + '/'
	tlist = glob.glob(tiffPath + '*.tif')
	tlist = sorted(tlist)
	
	if len(tlist) == 0:
		return False 
	
	remove_directory(zpath)

	store = zarr.DirectoryStore(zpath, dimension_separator='/')
	root = zarr.group(store=store, overwrite=True)
	data = root.create_group('muse')
	
	z = 0	
	for t in tlist:
		stack_size = os.path.getsize(t)
		z += np.floor(stack_size/bytes_per_image) + 1
	image = tiffio.imread(t, key = 0)
	
	x, y = image.shape
	
	zshape, zchunk = shape_definer(z,x,y,1)
	full = data.zeros('stitched', shape=zshape, chunks=zchunk, dtype="i2" )
	
	zcount = 0
	for t in tlist:
		stack_size = os.path.getsize(t)
		c = np.floor(stack_size/bytes_per_image) + 1
		for i in range(c):
			image = tiffio.imread(t, key = i)
			full[zcount] = image
			zcount += 1
	zarr.save(zpath, data)
	return True
	
	
		














































