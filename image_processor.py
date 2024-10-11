import warnings
warnings.filterwarnings("ignore")
import sys, os, getpass, shutil, glob, zarr, json, re, ast
import dask.array as da
import numpy as np
import cv2 as cv
from tqdm import tqdm
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from scipy.signal import savgol_filter
import skimage as sk
import scipy as sp
import matplotlib.pyplot as plt
import tifffile as tiffio

#Define Basic Variables Needed For the program
cmdInputs = {
	'-8b':{"name":"Downgrade Sample","types":[],"names":[],"variable":[],"active":False,"tooltips":"Downsamples zarr to 8-bit data"},
	'-br':{"name":"Brightness Increase","types":['int'],"names":['intensity'],"variable":[1000],"active":False,"tooltips":"Changes the global brightness of the images, Default: 1000"},
	'-bk':{"name":"BreakPoint","types":['int'],"names":['Index'],"variable":[1],"active":False,"tooltips":"Only take out one image as a PNG and ignores all others"},
	'-bt':{"name":"Black Hate  Contrasting","types":['int'],"names":['kernel'],"variable":[30],"active":False,"tooltips":"Performs Black Hat and Top Hate Contrasting, kernel size can be set"},
	'-ma':{"name":"Modality Add","types":['str','int'],"names":['type','kernel'],"variable":['None',30],"active":False,"tooltips":"Choose a modality to add to the image, often paired with -ms to subtract another modality. Of the form Modality type: Opening, Closing, Gradient, TopHat,BlackHat, and a kernal size Default: None, 30 pixels"},
	'-ms':{"name":"Modality Subtract","types":['str','int'],"names":['type','kernel'],"variable":['None',30],"active":False,"tooltips":"Choose a modality to subtract from the image, often paired with -ma to add another modality. Of the form Modality type: Opening, Closing, Gradient, TopHat,BlackHat, and a kernal size Default: None, 50 pixels"},
	'-d':{"name":"Dialation","types":['int'],"names":['kernel'],"variable":[2],"active":False,"tooltips":"Dialates image features. Default: 2 pixels"},
	'-e':{"name":"Erosion","types":['int'],"names":['kernel'],"variable":[2],"active":False,"tooltips":"Erodes image features, good for elucidating axons. Default: 2 pixels"},
	'-png':{"name":"PNG Stack","types":[],"names":[],"variable":[],"active":False,"tooltips":"Saves final output as PNG file"},
	'-c':{"name":"Finish Data","types":['list'],"names":['arrays'],"variable":[[]],"active":False,"tooltips":"Surveys data, collects all metadata, and outputs intial files"},
	'-f':{"name":"Flythrough","types":[],"names":[],"variable":[],"active":False,"tooltips":"Creates a video flythrough (Linux Only)"},
	'-fg':{"name":"Flythrough Grid","types":['int'],"names":["spacing"],"variable":[500],"active":False,"tooltips":"Creates a video flythrough with gridlines overlayed"},
	'-fi':{"name":"Crop and Histogram","types":['list'],"names":["crop points"],"variable":[[0,-1,0,-1]],"active":False,"tooltips":"Crops and histogram matches all data after cropping, this accepts a list [Height Min, Highe Max, Width Min, Width Max]"},
	'-i':{"name":"Individual Curves","types":[],"names":[],"variable":[],"active":False,"tooltips":"Saves the invidual curves for means, variance, and difference"},
	'-p':{"name":"Process Image","types":[],"names":[],"variable":[],"active":False,"tooltips":"Process Image"},	
	'-o':{"name":"Override Output","types":['str'],"names":['path'],"variable":["./output/"],"active":False,"tooltips":"Changes output directory"},	
	'-of':{"name":"Focus Bound","types":['int'],"names":['min'],"variable":[15],"active":False,"tooltips":"Sets boundaries for focus exclusion"},
	'-os':{"name":"Similarity Bounds","types":['int','int'],"names":["min","max"],"variable":[15,100],"active":False,"tooltips":"Sets boundary for similarity exclusion"},
	'-r':{"name":"Recursive","types":[],"names":[],"variable":[],"active":False,"tooltips":"Finds all folders in the base directory"},
	'-s':{"name":"Survey Data","types":[],"names":[],"variable":[],"active":False,"tooltips":"Surveys data, collects all metadata, and outputs intial files"},
	'-sb':{"name":"Scale Bar","types":[],"names":[],"variable":[],"active":False,"tooltips":"Adds scalebar to the output image of processor."},
	'-sk':{"name":"Skip Alignment","types":['list'],"names":['alignments'],"variable":[[]],"active":False,"tooltips":"Skips aligning between runs, assumes 0 shift for all shifts not entered. Enter shift in form of xshift,yshift for each array"},
	'-su':{"name":"Data Survey","types":[],"names":[],"variable":[],"active":False,"tooltips":"Rewrites Data Surveyor Metadata Files"},
	'-tr':{"name":"Truncate Data","types":['int'],"names":['amount'],"variable":[0],"active":False,"tooltips":"Rewrites Data Surveyor Metadata Files"},
	'-w':{"name":"Windowing","types":['int','int'],"names":["Min","Max"],"variable":[0,4095],"active":False,"tooltips":"Windows data and requires variables of the form <Min Intensity> <Max Intensity>"}
	}

#Define the directory structure which the program will use. Version 2.1 will stop wiping all previous data to go with efficency
dataFolder = 'data/'
metaFolder = 'metadata/'
oldFolder = metaFolder + 'history/'
pngFolder = dataFolder + 'png/'
surveyFolder = metaFolder + 'survey/'
movieFolder = metaFolder + 'movies/'


dataQualtyCheck = {
	"Mean Intensity":"means",
	"Laplacian Variance":"focus",
	"Image Difference":"difference"
	}

dataConversionTags = ['means','focus','difference','histogram','shift']

fly_font = cv.FONT_HERSHEY_SIMPLEX
fly_color = (255,255,255)
fly_thickness = 2
fly_font_scale = 1
fly_index_x = -50
fly_index_y = 50
fly_scale = 5

scalebar_length_pixels = 200
scalebar_index_x = -50
scalebar_index_y = -50 - scalebar_length_pixels
scalebar_length = int(scalebar_length_pixels * 0.9)
scalebar_color = (255,255,255)
scalebar_thickness = 5
scalebar_text_width = 118
scalebar_text_offset = (scalebar_length_pixels - scalebar_text_width) // 2

header_thickness = 10
header_font_scale = 3


byte_depth = np.power(2,12)
bytes_per_image = 24019288
logFileName = 'muse_application.log'

trim_length = 3

byteDeapth = 4096

elipse_size_align = 30
kernel_align = cv.getStructuringElement(cv.MORPH_ELLIPSE,(elipse_size_align,elipse_size_align))
align_image_size = 1500

modality_type = {
	'Opening':cv.MORPH_OPEN, 
	'Closing':cv.MORPH_CLOSE, 
	'Gradient':cv.MORPH_GRADIENT, 
	'TopHat':cv.MORPH_TOPHAT,
	'BlackHat':cv.MORPH_BLACKHAT
	}

#Baseline Functions
def printHelp():
	print("This is a help for Seckler Data Surveyor Software.")
	print("This is the first step in data processing and compiles all of the metadata for the MUSE software. It accepts the direct output from MUSE Acquire.")
	print("Command: python data_surveyor.py <File Name> <Path to Data> <Options>")
	print("")
	for entry in cmdInputs:
		print(generateHelpString(entry,cmdInputs[entry]))
	quit()

def generateHelpString(tag,entry):
	helpString = ''
	helpString += '-' + tag + ' '
	for e in entry['names']:
		helpString += '<' + e + '> '
	helpString += '		'
	helpString += entry['tooltips'] + ' '
	return helpString

def create_zarr_file(path,fname,x,y,z):
	zarr_path = path + fname + '.zarr'
	if os.path.isdir(zarr_path):
		shutil.rmtree(zarr_path)
	store = zarr.DirectoryStore(zarr_path, dimension_separator='/')
	root = zarr.group(store=store, overwrite=True)
	data = root.create_group('data')
	
	zshape, zchunk = shape_definer(z,x,y,1)
	zimg = data.zeros('0', shape=zshape, chunks=zchunk, dtype="i2" )
	create_zarr_attr(zarr_path)
	
	return zimg


def create_zarr_attr(zpath):
	dst = zpath + '/.zattrs'
	zattr = generate_zattr_file()
	
	zFile = open(dst,'w')
	
	for line in zattr:
		zFile.write(line + '\n')
	zFile.close()


def generate_zattr_file(depth = 12,pixel=0.9):
	x0 = float(pixel)
	y0 = float(pixel)
	x1 = pixel * 5
	y1 = pixel * 5
	x2 = pixel * 10
	y2 = pixel * 10

	zattr = [
		'{',
		'	"multiscales": [',
		'		{',
		'			"axes": [',
		'				{',
		'					"name": "z",',
		'					"type": "space",',
		'					"unit": "micrometer"',
		'				},',
		'				{',
		'					"name": "y",',
		'					"type": "space",',
		'					"unit": "micrometer"',
		'				},',
		'				{',
		'					"name": "x",',
		'					"type": "space",',
		'					"unit": "micrometer"',
		'				}',
		'			],',
		'			"datasets": [',
		'				{',
		'					"coordinateTransformations": [',
		'						{',
		'							"scale": [',
		f'								{depth},',
		f'								{x0},',
		f'								{y0}',
		'							],',
		'							"type": "scale"',
		'						}',
		'					],',
		'					"path": "data/0"',
		'				},',
		'				{',
		'					"coordinateTransformations": [',
		'						{',
		'							"scale": [',
		f'								{depth},',
		f'								{x1},',
		f'								{y1}',
		'							],',
		'							"type": "scale"',
		'						}',
		'					],',
		'					"path": "data/1"',
		'				},',
		'				{',
		'					"coordinateTransformations": [',
		'						{',
		'							"scale": [',
		f'								{depth},',
		f'								{x2},',
		f'								{x2}',
		'							],',
		'							"type": "scale"',
		'						}',
		'					],',
		'					"path": "data/2"',
		'				}',
		'			],',
		'			"name": "/data",',
		'			"version": "0.4"',
		'		}',
		'	]',
		'}',
	]
	return zattr


def contrast_enhance_for_image_align(image):
	contrast = 1.0
	mean = np.mean(image)
	image.astype('float')
	
	topHat = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel_align)
	blackHat = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel_align)
	contrast_image = image + topHat - blackHat
	
	
	contrast_image = image - mean
	contrast_image = contrast * contrast_image

	contrast_image = contrast_image + mean
	contrast_image = contrast_image.astype('uint8')
	
	return contrast_image


def coregister(img1,img2):
	shift, err, diff_phase = sk.registration.phase_cross_correlation(img1,img2)	
	img2 = sp.ndimage.shift(img2,shift)
	cv.imwrite('./img1.png',img1/16)
	cv.imwrite('./img2.png',img2/16)
	
	return img2, shift


def crop_to_center_of_image(img):
	width,height = img.shape
	mid_w = width // 2
	mid_h = height // 2
#	mid_h = 4 * align_image_size
	start_w = mid_w - align_image_size
	end_w = mid_w + align_image_size
	start_h = mid_h - align_image_size
	end_h = mid_h + align_image_size
	cropped = img[start_w: end_w, start_h: end_h]
	return cropped


def image_histogram(image, bitdepth = 4096):
	image_array = np.array(image)
	flattened_array = image_array.flatten()
	histogram, bin_edges = np.histogram(flattened_array, bins=bitdepth, range=(0, bitdepth))
	
	return histogram



def find_image_position(large_image, small_image):
	large_image = large_image // 16
	small_image = small_image // 16
	large_image = large_image.astype('uint8')
	small_image = small_image.astype('uint8')
		
	large_image = match_histograms(large_image, small_image)
	
	IMG1 = contrast_enhance_for_image_align(large_image)
	IMG2 = crop_to_center_of_image(small_image)
	IMG2 = contrast_enhance_for_image_align(IMG2)
		
	result = cv.matchTemplate(IMG1, IMG2, cv.TM_CCOEFF_NORMED)
	min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
	
	xShift = large_image.shape[0] // 2
	yShift = large_image.shape[1] // 2
	xShift -= align_image_size
	yShift -= align_image_size
	
	LOC = [max_loc[1] - xShift, max_loc[0] - yShift]
	
	return LOC

def find_image_position_(large_image, small_image):
	small_image = small_image - np.mean(small_image) + np.mean(large_image)
	
	IMG1 = contrast_enhance_for_image_align(large_image)
	IMG2 = crop_to_center_of_image(small_image)
	IMG2 = contrast_enhance_for_image_align(IMG2)
		
	result = cv.matchTemplate(IMG1, IMG2, cv.TM_CCOEFF_NORMED)
	min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
	
#	cv.imwrite('./pasted_img1.png',IMG1)
#	cv.imwrite('./pasted_img2.png',IMG2)
	
	xShift = large_image.shape[1] // 2
	yShift = large_image.shape[0] // 2
	xShift -= align_image_size
	yShift -= align_image_size
	
	print(min_val, max_val, min_loc, max_loc)
	
	LOC = [max_loc[0] - xShift, max_loc[1] - yShift]
	
	return LOC


def format_image_number_to_10000(c):
	if c < 10:
		text = '000' + str(c)
	elif c < 100:
		text = '00' + str(c)
	elif c < 1000:
		text = '0' + str(c)
	else:
		text = str(c)
	return text


def calculate_fwhm(gaussian_array):
	max_value = np.max(gaussian_array)
	half_max = max_value / 2.0
	indices_above_half_max = np.where(gaussian_array >= half_max)[0]
	fwhm = indices_above_half_max[-1] - indices_above_half_max[0]
	return fwhm


def logFileLoader(zarrPath):
	path = zarrPath + logFileName
	
	
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
	if os.path.isfile(path):
		rawFile = open(path, 'r')
	else:
		return {}
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
				try:
					y = float(m[i+1])
				except IndexError:
					y = float(0)
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
			try:
				hpanels = max(ys.values())
			except:
				hpanels = 1
			try:
				vpanels = max(xs.values())
			except:
				vpanels = 1
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


def match_histograms(image1, image2):
	hist_ref, bins = np.histogram(image2.flatten(), 256, [0, 256])
	cdf_ref = hist_ref.cumsum()
	cdf_ref_normalized = cdf_ref * hist_ref.max() / cdf_ref.max()  # Normalize
	hist_src, bins = np.histogram(image1.flatten(), 256, [0, 256])
	cdf_src = hist_src.cumsum()
	cdf_src_normalized = cdf_src * hist_src.max() / cdf_src.max()  # Normalize
	lut = np.zeros(256, dtype=np.uint8)
	g_j = 0
	for i in range(256):
		while g_j < 255 and cdf_src[i] > cdf_ref[g_j]:
			g_j += 1
		lut[i] = g_j
	
	# Apply the lookup table to map the pixel values of the source image
	image1_matched = cv.LUT(image1, lut)
	
	return image1_matched


def overlay_acu_and_index_number_on_image(image,zarrNumber,i):
	position = create_text_position(image.shape)
	if zarrNumber == "":
		text = f"Index#{i}"
	else:
		text = f"Acq#{zarrNumber}, Index#{i}"
	cv.putText(image, text, position, fly_font, fly_font_scale, fly_color, fly_thickness, cv.LINE_AA)
	return image

def overlay_image_attr_on_image(image,mean,laplace,difference):
	text = [f"Mean: {int(mean)}",f"Focus: {int(laplace)}",f"Similarity: {int(10000 * (1 - difference))}"]
	
	x, y = image.shape
	
	for i in range(len(text)):
		position = (50, 50 * i + 50)
		cv.putText(image, text[i], position, fly_font, fly_font_scale, fly_color, fly_thickness, cv.LINE_AA)
	return image

def overlay_scalebar_on_image(image,brate=1, length_mode=1):
	position = create_text_position(image.shape,False)
	
	
	scalebarColor = (brate * scalebar_color[0],brate * scalebar_color[1],brate * scalebar_color[2])
	
	start_point = position
	end_point = (position[0] + scalebar_length_pixels, position[1])
	cv.line(image, start_point, end_point, scalebarColor, scalebar_thickness)
	
	
	text = f"{scalebar_length} um"
	text_size = cv.getTextSize(text, fly_font, fly_font_scale, fly_thickness)[0]
	scalebar_text_offset = (scalebar_length_pixels - text_size[0]) // 2

	flyColor = (brate * fly_color[0],brate * fly_color[1],brate * fly_color[2])

	
	text_position = (start_point[0] + scalebar_text_offset, start_point[1] - 10)
	cv.putText(image, text, text_position, fly_font, fly_font_scale, flyColor, fly_thickness, cv.LINE_AA)
	return image


def overlay_header_on_image(image,fname,brate=1):
	text = fname.capitalize()
	text_size = cv.getTextSize(text, fly_font, header_font_scale, header_thickness)[0]
	
	flyColor = (brate * fly_color[0],brate * fly_color[1],brate * fly_color[2])
	
	image_width = image.shape[1]
	text_x = (image_width - text_size[0]) // 2
	text_y = 30 + text_size[1]  # 30 pixels padding from the top
	cv.putText(image, text, (text_x, text_y), fly_font, header_font_scale, flyColor, header_thickness, cv.LINE_AA)
	return image


def overlay_grid_lines_on_image(image,spacing):
	vLines = image.shape[0] // spacing
	hLines = image.shape[1] // spacing
	
	for h in range(hLines):
		cv.line(image, (spacing * (h + 1),spacing), (spacing * (h + 1),(vLines - 1) * spacing), scalebar_color, 2)
	for v in range(vLines):
		cv.line(image, (spacing,spacing * (v + 1)), ((hLines - 1) * spacing,spacing * (v + 1)), scalebar_color, 2)
	return image
	


def shape_definer(n,x,y,scale):
	zshape = (n,int(x / scale),int(y / scale))
	zchunk = (4,int(x / scale),int(y / scale))
	return zshape, zchunk


def create_text_position(shape,acq=True):
	if acq:
		x = fly_index_x
		y = fly_index_y
	else:
		x = scalebar_index_x
		y = scalebar_index_y

		
	if x < 0:
		x = shape[0] + x
	if y < 0:
		y = shape[1] + y
	return (y,x)





def findAllDir(path):
	flist = glob.glob(path + "*")
	
	allRuns = []
	for fname in flist:
		if os.path.isdir(fname):
			run = fname.split(os.path.sep)[-1]
			allRuns.append(run)
	
	allRuns = sorted(allRuns)
	return allRuns

def make_directory(directory):
	if not os.path.isdir(directory):
		os.makedirs(directory)

def remove_directory(directory):
	if os.path.isdir(directory):
		shutil.rmtree(directory)

def replace_directory(directory):
	if os.path.isdir(directory):
		shutil.rmtree(directory)
	os.makedirs(directory)


def crop_to_center(width,height):
	mid_w = width // 2
	mid_h = height // 2
	start_w = mid_w - 1500
	end_w = mid_w + 1500
	start_h = mid_h - 1500
	end_h = mid_h + 1500
	return start_w, end_w, start_h, end_h



def data_loader_from_json(metaPath):
	dataPath = metaPath + '/data.dat'
	if os.path.exists(dataPath):
		with open(dataPath) as user_file:
			file_contents = user_file.read()
		data = json.loads(file_contents)
		data = convertDataTagToArray(data)
		
	else:
		data = {}
	return data

def data_saver_to_json(datum,metaPath):
	dataPath = metaPath + '/data.dat'
	readyData = convertDataTagTolist(datum)
	
	with open(dataPath, 'w') as f:
		json.dump(readyData, f)
	
	datum = convertDataTagToArray(datum)
	return datum

def convertDataTagToArray(d):
	for tag in dataConversionTags:
		if tag in d:
			for i in d[tag]:
				d[tag][i] = np.array(d[tag][i])
	return d

def convertDataTagTolist(d):
	for tag in dataConversionTags:
		if tag in d:
			for i in d[tag]:
				d[tag][i] = d[tag][i].tolist()
	return d


def get_image_from_zarr_as_dask_array(path):
	try:
		zimg = da.from_zarr(path, component="muse/stitched/")
		return zimg
	except:
		if save_single_panel_tiff_as_zarr_file(path):
			zimg = da.from_zarr(path, component="muse/stitched/")
			return zimg
		else:
			print("Filename, "+path+" is corrupted or incorrect and did not produce a file from zarr file...")
			return None

def get_zarr_attr_as_dict(path):
	data = zarr.open(path, mode='r')
	try:
		return data.attrs['multiscales'][0]
	except:
		return {}

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
		c = int(np.floor(stack_size/bytes_per_image) + 1)
		for i in range(c):
			image = tiffio.imread(t, key = i)
			full[zcount] = image
			zcount += 1
	return True



class inputs:
	def __init__(self,cmd):
		self.cmdInputs = cmdInputs
		self.inputParser(cmd)
	
	def inputParser(self,cmd):
		n = len(cmd)
	
		if n < 3 and '-h' not in cmd:
			print("No filename and/or path given...")
			quit()
	
		if '-h' in cmd:
			printHelp()
	
		self.base_path = cmd[2]
	
		for i in range(n):
			tag = cmd[i]
			if tag[0] == "-" and tag in self.cmdInputs:
				self.cmdInputs[tag]['active'] = True
				
				m = len(self.cmdInputs[tag]['names'])
				for j in range(m):
					try:
						inputValue = cmd[i + j + 1]
						if self.cmdInputs[tag]['types'][j] == 'float':
							inputValue = float(inputValue)
						elif self.cmdInputs[tag]['types'][j] == 'str':
							if inputValue[0] == '-':
								self.cmdInputs[tag]['active'] = False
								print(f"Input {tag} has failed to read in input values, using defaults...")	
						elif self.cmdInputs[tag]['types'][j] == 'list':
							inputValue = ast.literal_eval(inputValue)
						else:
							inputValue = int(inputValue)
						self.cmdInputs[tag]['variable'][j] = inputValue
					except:
						print(f"Input {tag} has failed to read in input values, using defaults...")
			elif tag == '-jms':
				self.base_path = '/media/' + getpass.getuser() + '/' + cmd[2] + '/data/'



class globals:
	def __init__(self,fname,PATH,cmd):
		print("Setting up all global variables and validating data...")
		self.fname = fname
		self.base_path = PATH
		self.cmdInputs = cmd
		self.check_output_directory_structure_and_load_all_metadata()
		self.loadMetadata()
		self.findAllValidAcq()
		self.scrape_tiff_metadata()
		
		if self.cmdInputs['-s']['active']:
			print(f"Conducting a survey of {fname}, this may take several minutes...")
			self.survey_all_data_and_compile_databases()
		if self.cmdInputs['-c']['active']:
			print(f"Finishing preprocessing on {fname}, this may take several minutes...")
			self.finish_data_preprocessing()
		if self.cmdInputs['-fi']['active']:
			self.histogram_map_and_crop()
		if self.cmdInputs['-p']['active']:
			self.process_all_images()
	

	def check_output_directory_structure_and_load_all_metadata(self):
		if self.cmdInputs['-o']['active']:
			self.outpath = cmdInputs['-o']['variable'][0]
		else:
			self.outpath = './output/'
		
		if not self.outpath.endswith((self.fname + '/')):
			self.outpath = self.outpath + self.fname + '/'
	
		self.inPath = self.base_path + self.fname + '/'
		self.metaPath = self.outpath + metaFolder
		self.oldPath = self.outpath + oldFolder
		self.imgPath = self.outpath + dataFolder
		self.pngPath = self.outpath + pngFolder
		self.surveyPath = self.outpath + surveyFolder
		self.moviePath = self.outpath + movieFolder
		
		#Sets up directory structure and wipes old data
		make_directory(self.outpath)
		make_directory(self.metaPath)
		make_directory(self.oldPath)
		make_directory(self.surveyPath)
		make_directory(self.imgPath)
		make_directory(self.moviePath)
		
	
	def saveMetadata(self):
		self.data = data_saver_to_json(self.data,self.metaPath)
	
	
	def loadMetadata(self):
		self.data = data_loader_from_json(self.metaPath)
	
	
	def loadZarrFile(self,zarrNumber):
		zpath = self.inPath + 'MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
		img = get_image_from_zarr_as_dask_array(zpath)
		return img
	
	def getAttrFromZarr(self,zarrNumber):
		zpath = self.inPath + 'MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
		self.data['attr'][zarrNumber] = get_zarr_attr_as_dict(zpath)
	
	def get_log_file_run(self,zarrNumber):
		for r in self.data['log']['runList']:
			run = r.split(' ')[-1]
			if run == zarrNumber:
				return r
		return None

	def findAllValidAcq(self):
		flist = glob.glob(self.inPath + "*.zarr")
		
		allRuns = []
		for fname in flist:
			run = fname.split('.')[0]
			run = run.split('_')[-1]
			allRuns.append(run)
		
		self.allArrays = []
		
		for zarrNumber in allRuns:
			img = self.loadZarrFile(zarrNumber)
			try:
				img = np.array(img[0])
				self.allArrays.append(int(zarrNumber))
			except:
				pass
		self.allArrays = sorted(self.allArrays)
		#Added to prevent error when more than 10 acq are in
		for i in range(len(self.allArrays)):
			self.allArrays[i] = str(self.allArrays[i])
	
	def scrape_tiff_metadata(self):
		self.tiffData = {}
		
		for zarrNumber in self.allArrays:
			self.tiffData[zarrNumber] = {}
			tiffPath = self.inPath + 'MUSE_acq_' + zarrNumber + '/'
			tlist = glob.glob(tiffPath + '*.tif')
			tlist = sorted(tlist)
			
			z = 0
			count = 0
			for t in tlist:
				self.tiffData[zarrNumber][count] = {'path':t}
				stack_size = os.path.getsize(t)
				z += np.floor(stack_size/bytes_per_image) + 1
				self.tiffData[zarrNumber][count]['images'] = z
		
	
	
	#START OF SURVEYOR FUNCTIONS
	def survey_all_data_and_compile_databases(self):
		passer = False
		if 'attr' not in self.data:
			self.scrap_all_attr_files_from_zarrs()
		if 'means' not in self.data:
			passer = self.calculate_global_means_for_all_images_and_save_attributes()
		if 'focus' not in self.data:
			passer = self.calculate_global_focus_for_all_images()
		if 'difference' not in self.data:
			passer = self.calculate_global_adjacent_image_difference_for_all_images()
		if 'histogram' not in self.data:
			passer = self.calculate_histogram_for_all_data()
		if 'height' not in self.data or 'width' not in self.data:
			passer = self.determine_pixel_dimensions_for_all_images()
		if 'log' not in self.data:
			self.data['log'] = logFileLoader(self.inPath)
		if passer or self.cmdInputs['-su']['active']:
			self.rewrite_data_survey_file_and_write_survey_images()
		
		if self.cmdInputs['-f']['active']:
			self.create_flythrough_video_for_data_visualization()
		if self.cmdInputs['-i']['active']:
			self.save_inidividual_curves()
		self.data["stage1_flags"] = self.cmdInputs
		self.saveMetadata()


		
		
	def scrap_all_attr_files_from_zarrs(self):
		self.data['attr'] = {}
		for zarrNumber in self.allArrays:
			self.getAttrFromZarr(zarrNumber)
		self.saveMetadata()
	
	
	def calculate_global_means_for_all_images_and_save_attributes(self):
		print('Calculating Global Means for all images')
		self.data['means'] = {}
		self.data['length'] = {}
		self.data['width'] = {}
		self.data['height'] = {}
		for zarrNumber in tqdm(self.allArrays):
			img = self.loadZarrFile(zarrNumber)
			self.data['means'][zarrNumber] = np.zeros(img.shape[0])
			self.data['width'][zarrNumber] = img.shape[1]
			self.data['height'][zarrNumber] = img.shape[2]
			
			for i in range(img.shape[0]):
				self.data['means'][zarrNumber][i] = np.mean(img[i])
				self.data['length'][zarrNumber] = i
				if self.data['means'][zarrNumber][i] == 0:
					break
		self.saveMetadata()
		return True


	def calculate_global_focus_for_all_images(self):
		print('Calculating global focus for all images')
		self.data['focus'] = {}
		
		for zarrNumber in tqdm(self.allArrays):
			img = self.loadZarrFile(zarrNumber)
			self.data['focus'][zarrNumber] = np.zeros(img.shape[0])
			
			width, height = img[0].shape
			start_w, end_w, start_h, end_h = crop_to_center(width,height)
			
			for i in range(img.shape[0]):
				image = np.array(img[i][start_w:end_w,start_h:end_h])
				
				vimage = image - np.mean(image)
				vimage = 3.0 * vimage
				vimage = vimage + 2027
				vimage = np.clip(vimage,0,4095)
		
				blurred_image = cv.GaussianBlur(vimage, (15,15), 0)
				variance = cv.Laplacian(blurred_image, cv.CV_64F)
				self.data['focus'][zarrNumber][i] = variance.var()
				if self.data['means'][zarrNumber][i] == 0:
					break
		self.saveMetadata()
		return True
	
	
	def calculate_global_adjacent_image_difference_for_all_images(self):
		print('Calculating Adjacent Image Differences For All Images')
		self.data['difference'] = {}
		for zarrNumber in tqdm(self.allArrays):
			img = self.loadZarrFile(zarrNumber)
			self.data['difference'][zarrNumber] = np.zeros(img.shape[0])
			
			width, height = img[0].shape
			start_w, end_w, start_h, end_h = crop_to_center(width,height)	
			
			for i in range(img.shape[0]):
				pImage = np.array(img[i-1][start_w:end_w,start_h:end_h])
				image = np.array(img[i][start_w:end_w,start_h:end_h])
				similarity_score, _ = ssim(image, pImage, full=True)
				self.data['difference'][zarrNumber][i] = similarity_score
				if self.data['means'][zarrNumber][i] == 0:
					break
		self.saveMetadata()
		return True
	

	def calculate_histogram_for_all_data(self):
		print('Calculating Histograms for all images')
		self.data['histogram'] = {}
		for zarrNumber in tqdm(self.allArrays):
			img = self.loadZarrFile(zarrNumber)
			self.data['histogram'][zarrNumber] = []
			
			for i in range(img.shape[0]):
				self.data['histogram'][zarrNumber].append(image_histogram(img[i]))
				if self.data['means'][zarrNumber][i] == 0:
					break
			self.data['histogram'][zarrNumber] = np.array(self.data['histogram'][zarrNumber])
		
		self.saveMetadata()
		return True

	def determine_pixel_dimensions_for_all_images(self):
		self.data['width'] = {}
		self.data['height'] = {}
		for zarrNumber in self.allArrays:
			img = self.loadZarrFile(zarrNumber)
			self.data['width'][zarrNumber] = img.shape[1]
			self.data['height'][zarrNumber] = img.shape[2]
		self.saveMetadata()
		return True


	def rewrite_data_survey_file_and_write_survey_images(self):
		print('Writing Log File and saving a selection of images from all zarr acquisitions')
		path = self.metaPath + "/quality.csv"
		logFILE = open(path,'w')
		
		now = datetime.now()
		date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
		
		logFILE.write(f"Quality Check Generated on {date_time_str} by {getpass.getuser()}\n")
		
		logFILE.write('Acquistion,# of Images,Width,Height,')
		for tag in dataQualtyCheck:
			logFILE.write(tag + ',,')
		logFILE.write('\n')
		for zarrNumber in self.allArrays:
			logFILE.write(zarrNumber + ',')
			logFILE.write(str(int(self.data['length'][zarrNumber])) + ',')
			logFILE.write(str(int(self.data['width'][zarrNumber])) + ',')
			logFILE.write(str(int(self.data['height'][zarrNumber])) + ',')
			for tag in dataQualtyCheck:
				logFILE.write(self.prepareDataForSurvey(dataQualtyCheck[tag],zarrNumber) + ',')
			logFILE.write('\n')
			self.save_averaged_histograms(zarrNumber,self.data['histogram'][zarrNumber])
			self.saveSurveyImage(zarrNumber)
		logFILE.close()
	

	def prepareDataForSurvey(self,tag,zarrNumber):
		pData = np.mean(self.data[tag][zarrNumber][self.data[tag][zarrNumber] != 0])
		try:
			if tag == 'difference':
				pData = 1000 - 1000 * pData
			pData = int(pData)
		except ValueError:
			pData = 0
		sData = np.std(self.data[tag][zarrNumber][self.data[tag][zarrNumber] != 0])
		try:
			if tag == 'difference':
				sData = 1000 - 1000 * sData
			sData = int(sData)
		except ValueError:
			sData = 0
		
		STRING_TO_RETURN = str(pData) + ',' + str(sData)
		return STRING_TO_RETURN
	
	
	def save_averaged_histograms(self,zarrNumber,histogram):
		hist = np.mean(histogram,axis=0)
		hist[0] = 0
		hist[4050:-1] = 0
		x = np.arange(4096)
		plt.figure(figsize=(10, 6))
		plt.plot(x, hist)  # bin_edges has one extra element
		plt.title(f"Histogram of Pixel Intensities for Acq #{zarrNumber}")
		plt.xlabel("Pixel Intensity")
		plt.ylabel("Frequency")
		plt.savefig(self.surveyPath + f'histogram_{zarrNumber}.png')
		plt.close()

	def save_individual_histograms(self,path,hist,index):
		hist[0] = 0
		hist[4050:-1] = 0
		x = np.arange(4096)
		plt.figure(figsize=(10, 6))
		plt.plot(x, hist)  # bin_edges has one extra element
		plt.title(f"Histogram of Pixel Intensities for Acq #{index}")
		plt.xlabel("Pixel Intensity")
		plt.ylabel("Frequency")
		plt.savefig(path + f'histogram_{index}.png')
		plt.close()
	
	
	def saveSurveyImage(self,zarrNumber):
		img = self.loadZarrFile(zarrNumber)
		index = int(self.data['length'][zarrNumber]) -1
		image = np.array(img[index])
		
		cv.imwrite(self.surveyPath + f"example_{zarrNumber}_last.png",image/16)
		
		image = np.array(img[0])
		cv.imwrite(self.surveyPath + f"example_{zarrNumber}_first.png",image/16)
	
	
	def create_flythrough_video_for_data_visualization(self):
		self.tmpPath = self.moviePath + '/tmp/'
		flythroughs = self.determine_number_of_flythroughs_to_make()
		
		for fly in tqdm(flythroughs):
			path = self.moviePath + f'flythrough_{fly}.mp4'
			if os.path.isfile(path):continue
			FlyArrays = flythroughs[fly]
			replace_directory(self.tmpPath)
			counter = 0
			for zarrNumber in FlyArrays:
				img = self.loadZarrFile(zarrNumber)
				
				for i in range(self.data['length'][zarrNumber]+1):
					self.write_tmp_image(img[i],i,counter,zarrNumber)
					counter += 1
			self.compile_pngs_to_movie(fly)
			remove_directory(self.tmpPath)
	
	def create_final_flythrough_video_for_data_visualization(self,gridline = False):
		if gridline:
			path = self.moviePath + 'grid_flythrough.mp4'
		else:
			path = self.moviePath + 'raw_flythrough.mp4'
		self.tmpPath = self.moviePath + 'tmp/'
		replace_directory(self.tmpPath)
		
		if os.path.isfile(path):
			os.remove(path)
		for i in range(self.zimg.shape[0]):
			self.write_tmp_image(self.zimg[i],i,i,grid=gridline)
		self.compile_pngs_to_movie()
		remove_directory(self.tmpPath)
	
	
	def write_tmp_image(self,image,i,counter,zarrNumber='',grid=False):
		image = np.array(image)
		image = image / 16
		resolution = (int(image.shape[1] / fly_scale), int(image.shape[0] / fly_scale))
		image = cv.resize(image, resolution, interpolation= cv.INTER_LINEAR)
		
		if grid:
			image = overlay_grid_lines_on_image(image,self.cmdInputs['-fg']['variable'][0] // fly_scale)
		image = overlay_acu_and_index_number_on_image(image,zarrNumber,i)
		image = overlay_scalebar_on_image(image)
		image = overlay_header_on_image(image,self.fname)
		
		if self.cmdInputs['-s']['active']:
			image = overlay_image_attr_on_image(image,self.data['means'][zarrNumber][i],self.data['focus'][zarrNumber][i],self.data['difference'][zarrNumber][i])
		
		
		
		c = format_image_number_to_10000(counter)
		cv.imwrite(self.tmpPath + f"image_{c}.png",image)
		


	def determine_number_of_flythroughs_to_make(self):
		flythrough_number = 0
		flyThroughRuns = {}
		width = 0
		height = 0
		flyThroughRuns[flythrough_number] = []
		
		for zarrNumber in self.allArrays:
			if len(flyThroughRuns[flythrough_number]) == 0 or (width == self.data['width'][zarrNumber] and height == self.data['height'][zarrNumber]):
				flyThroughRuns[flythrough_number].append(zarrNumber)
			else:
				flythrough_number += 1
				flyThroughRuns[flythrough_number] = []
				flyThroughRuns[flythrough_number].append(zarrNumber)
			width = self.data['width'][zarrNumber]
			height = self.data['height'][zarrNumber]
		return flyThroughRuns
	
	def save_inidividual_curves(self):
		self.write_curve_file('difference')
		self.write_curve_file('means')
		self.write_curve_file('focus')
		
	
	def write_curve_file(self,name):
		curvesFile = open(self.metaPath + name + ".csv", 'w')
		length = 0
		
		
		for zarrNumber in self.allArrays:
			if self.data[name][zarrNumber].shape[0] > length:
				length = self.data[name][zarrNumber].shape[0]
			curvesFile.write(f'Acq#{zarrNumber},')
		curvesFile.write('\n')
		
		
		for i in range(length):
			for zarrNumber in self.allArrays:
				if i < self.data[name][zarrNumber].shape[0]:
					datum = self.data[name][zarrNumber][i]
					curvesFile.write(f"{datum},")
				else:
					curvesFile.write(f'0,')
			curvesFile.write('\n')
		curvesFile.close()
	
	def compile_pngs_to_movie(self,runNumber=""):
		if runNumber == "":
			mName = 'raw_flythrough'
		else:
			mName = f'flythrough_{runNumber}'
		cmd = f'ffmpeg -framerate 10 -i {self.tmpPath}image_%04d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -r 30 -y -pix_fmt yuv420p {self.moviePath}{mName}.mp4'
		stream = os.popen(cmd)
		output = stream.read()
		print(output)
	
	
	#BEGIN COMPILING DATA HERE
	def finish_data_preprocessing(self):
		print("Gathering information about compiled acquisitions...")
		self.data['acqQuality'] = self.cmdInputs['-c']['variable'][0]
		
		#Sort Acquisitions by Input
		if len(self.allArrays) != len(self.data['acqQuality']):
			print("Error: Acquisition Tags must be the same length as all Acquisition in raw zarr file. Please adjust input...")
			return
		if self.sort_all_arrays_by_quality():
			return
		
		#Salavge misaligned Acquisitions
		if len(self.svgArray) > 0:
			print("Restitching salvagable data...")
			self.salvage_all_salvageable_images()

		if self.cmdInputs['-sk']['active']:
			self.read_in_shifts()
		else:
			self.coregister_between_blocks()
		
		self.compile_images_into_single_zarr()
		
#		#Check if Flythrough and write new flythrough
		if self.cmdInputs['-f']['active'] or self.cmdInputs['-f']['active']:
			print("Creating final flythrough of raw data...")
			self.create_final_flythrough_video_for_data_visualization()
		if self.cmdInputs['-fg']['active']:
			print("Creating gridline flythrough of raw data...")
			self.create_final_flythrough_video_for_data_visualization(True)
		
		self.data['shift'] = self.shift
		self.data["stage2_flags"] = self.cmdInputs
		self.saveMetadata()
		

	def sort_all_arrays_by_quality(self):
		self.useArray = []
		self.badArray = []
		self.ignArray = []
		self.svgArray = []
		
		n = len(self.allArrays)
		
		for i in range(n):
			if self.data['acqQuality'][i] == 1:
				self.useArray.append(self.allArrays[i])
			elif self.data['acqQuality'][i] == 2:
				self.badArray.append(self.allArrays[i])
			elif self.data['acqQuality'][i] == 3:
				self.ignArray.append(self.allArrays[i])
			elif self.data['acqQuality'][i] == 4:
				self.svgArray.append(self.allArrays[i])
			else:
				print("Error: Acquisition Tags must be all integers between 1 and 4. Please adjust input...")
				return True
		return False
	
	def load_tiff(self,zarrNumber,index):
		tiffPath = self.inPath + 'MUSE_acq_' + zarrNumber + '/'
		
		for run in self.tiffData[zarrNumber]:
			if index < self.tiffData[zarrNumber][run]['images']:
				path = self.tiffData[zarrNumber][run]['path']
				break
			else:
				index -= self.tiffData[zarrNumber][run]['images']
		image = tiffio.imread(path, key = index)
		return image
	
		
	def salvage_all_salvageable_images(self):
#		self.svgArray
		pass
		#Write Code to:
		#1) input seed image, self.metaPath + '/align/image_X.png'
			#1A) Look for seed image
			#1B) If not found will to last valid array, self.useArray
			#1C If neither, throw error
		#2) Read in Panels and relative locations, self.data['log']
		#3) Read in Tiff File, use self.load_tiff(<run number>,<tiff index>) to load a single image from a tiff. Tiffs are huge and we'll crash the program if we try to load the whole thing all at once
		#4) Find Panel from log to Image in TIFF
		#5) Create Zarr File to replace old zarr, rename old zarr just in case we want to keep it
		#6) Use seed image to align where panels go to form new image
		#7) Write new image to zarr
		#8) Repeat
		#9) All elements from self.svgArray need to be added to self.useArray and sorted into proper order self.useArray = sorted(self.useArray)
		
		#Look to SR015-CR2-3 and SR015-CR2-2 for example data. You can find it on the Shoffstall NAS under source-data/SR015/<foldername>
	
	def read_in_shifts(self):
		rawShifts = self.cmdInputs['-sk']['variable']
		n = len(self.useArray)
		self.shift = {}
		
		for i in range(n):
			zarrNumber = self.useArray[i]
			try:
				self.shift[zarrNumber] = np.array([rawShifts[2*i],rawShifts[2*i_1]])
			except:
				self.shift[zarrNumber] = np.array([0,0])
		
		self.clean_up_shift_array()
		
		self.width = 0
		self.height = 0
		self.length = 0
		for zarrNumber in self.useArray:
			if self.width < self.data['width'][zarrNumber]:
				self.width = self.data['width'][zarrNumber]
			if self.height < self.data['height'][zarrNumber]:
				self.height = self.data['height'][zarrNumber]
			self.length += self.data['length'][zarrNumber]

			

	def coregister_between_blocks(self):
#		This function needs to look at the first/last valid image of each run and coregister them together recording the relative shifts between these runs in self.shift.
#		self.shift is of the form self.shift[run number] = np.array([x_shift,y_shift]). 
#		This will assume Integer Shift
#		Use the variable self.oldPath to look for masks.

		self.shift = {}
		
		#Feel Free to Delete Everything After this
		
		
		self.width = 0
		self.height = 0
		self.length = 0
		for zarrNumber in self.useArray:
			if self.width < self.data['width'][zarrNumber]:
				self.width = self.data['width'][zarrNumber]
			if self.height < self.data['height'][zarrNumber]:
				self.height = self.data['height'][zarrNumber]
			self.length += self.data['length'][zarrNumber]
		
		print(self.width,self.height)		

		for zarrNumber in self.useArray:
			if self.width == self.data['width'][zarrNumber] and self.height == self.data['height'][zarrNumber]:
				starting_zarr = zarrNumber
				break
		
		start_pos = "Start"
		if starting_zarr != self.useArray[0]:
			zarrNumber = self.useArray[-1]
			if self.width == self.data['width'][zarrNumber] and self.height == self.data['height'][zarrNumber]:
				start_pos = "End"
				starting_zarr = zarrNumber
			else:
				start_pos = "Middle"
		
		self.shift[starting_zarr] = np.array([0,0])
		
		INDEX = self.useArray.index(starting_zarr)
		
		
		if start_pos == "Start":
			self.coregister_forward_from_index(INDEX)
		elif start_pos == "Middle":
			self.coregister_forward_from_index(INDEX)
			self.coregister_backward_from_index(INDEX)
		elif start_pos == "End":
			self.coregister_backward_from_index(INDEX)
		
		self.clean_up_shift_array()
		
		arrMin_x = 0
		arrMax_x = 0
		arrMin_y = 0
		arrMax_y = 0
		for zarrNumber in self.useArray:
			if self.shift[zarrNumber][0] < arrMin_x:
				arrMin_x = self.shift[zarrNumber][0]
			elif self.shift[zarrNumber][0] > arrMax_x:
				arrMax_x = self.shift[zarrNumber][0]
			if self.shift[zarrNumber][1] < arrMin_y:
				arrMin_y = self.shift[zarrNumber][1]
			elif self.shift[zarrNumber][1] > arrMax_y:
				arrMax_y = self.shift[zarrNumber][1]
		
		self.width = self.width + arrMax_x - arrMin_x
		self.height = self.height + arrMax_y - arrMin_y
		
		arrShift = np.array([-arrMin_x,-arrMin_y])
		for zarrNumber in self.useArray:
			self.shift[zarrNumber] += arrShift
		
		if self.width % 2 != 0:
			self.width += 1
		if self.height % 2 != 0:
			self.height += 1
		
	def clean_up_shift_array(self):
		n = len(self.useArray)
		for i in range(n):
			if i > 0:
				izarrNumber = self.useArray[i-1]
				zarrNumber = self.useArray[i]
				self.shift[zarrNumber] = self.shift[zarrNumber] + self.shift[izarrNumber]
		
	
	def coregister_forward_from_index(self,Index):
		n = len(self.useArray) - Index - 1
		
		for i in range(n):
			iIndex = i + Index
			fIndex = i + Index + 1
			self.coregister_between_two_blocks(iIndex,fIndex)
	
	
	def coregister_backward_from_index(self,Index):
		n = Index
		
		for i in range(n):
			iIndex = Index - i
			fIndex = Index - i - 1
			self.coregister_between_two_blocks(iIndex,fIndex)
			
	
	
	def coregister_between_two_blocks(self,first_index,second_index):
		zarrNumber_1 = self.useArray[first_index]
		zarrNumber_2 = self.useArray[second_index]
		if first_index > second_index:
			offset_1 = 0
			offset_2 = self.data['length'][zarrNumber_2]
		else:
			offset_1 = self.data['length'][zarrNumber_1]
			offset_2 = 0
		
		zimg1 = np.array(self.loadZarrFile(zarrNumber_1)[offset_1])
		zimg2 = np.array(self.loadZarrFile(zarrNumber_2)[offset_2])
		
		cv.imwrite(f'./{zarrNumber_2}_img1.png',zimg1//16)
		cv.imwrite(f'./{zarrNumber_2}_img2.png',zimg2//16)

		
		self.shift[zarrNumber_2] =  find_image_position(zimg1, zimg2)
		
	
	def compile_images_into_single_zarr(self):
		index = 0
		for zarrNumber in tqdm(self.useArray):
			zimg = self.loadZarrFile(zarrNumber)
			z = self.data['length'][zarrNumber]
			for i in range(z):
				MEAN = self.data['means'][zarrNumber][i]
				FOCUS = self.data['focus'][zarrNumber][i]
				SSIM = int(10000 * (1 - self.data['difference'][zarrNumber][i]))
				try:
					pSSIM = int(10000 * (1 - self.data['difference'][zarrNumber][i+1]))
				except:
					pSSIM = int(10000 * (1 - self.data['difference'][zarrNumber][i]))
				if MEAN > 0 and FOCUS > self.cmdInputs['-of']['variable'][0] and SSIM > self.cmdInputs['-os']['variable'][0] and SSIM < self.cmdInputs['-os']['variable'][1] and pSSIM < self.cmdInputs['-os']['variable'][1]:
					index += 1

		self.zimg = create_zarr_file(self.imgPath,f"rawData",self.width,self.height,index)
		index = 0
		for zarrNumber in tqdm(self.useArray):
			zimg = self.loadZarrFile(zarrNumber)
			z = self.data['length'][zarrNumber]
			for i in range(z):
				MEAN = self.data['means'][zarrNumber][i]
				FOCUS = self.data['focus'][zarrNumber][i]
				SSIM = int(10000 * (1 - self.data['difference'][zarrNumber][i]))
				try:
					pSSIM = int(10000 * (1 - self.data['difference'][zarrNumber][i+1]))
				except:
					pSSIM = int(10000 * (1 - self.data['difference'][zarrNumber][i]))
				if MEAN > 0 and FOCUS > self.cmdInputs['-of']['variable'][0] and SSIM > self.cmdInputs['-os']['variable'][0] and SSIM < self.cmdInputs['-os']['variable'][1] and pSSIM < self.cmdInputs['-os']['variable'][1]:
					self.zimg[index] = self.preprocess(zimg[i],self.shift[zarrNumber])
					index += 1
		
		
		
		
	def preprocess(self,image,shift):
		#Add code in here to check if the image is actually useable and return none if not
		image = np.array(image)
		
		histogram = image_histogram(image)
		smooth_signal = savgol_filter(histogram, 50, 2)
		smooth_signal[0:100] = 0
		smooth_signal[4050:-1] = 0
		
		darkpeak = np.argmax(smooth_signal)
		
		image = image.astype(float)
		
#		image[image != 0] -= int(mean)
#		image[image != 0] += self.mean
		image = np.clip(image,0,byte_depth-1)
		
		
		image = self.add_shifted_arrays(image, shift)
		return image
	
	def add_shifted_arrays(self,image, shift):
		result = np.zeros((self.width,self.height))
		
		xStart = shift[0]
		yStart = shift[1]
		xEnd = xStart + image.shape[0]
		yEnd = yStart + image.shape[1]
		
		result[xStart:xEnd,yStart:yEnd] += image
		return result

	
	def histogram_map_and_crop(self):
		#STAGE 3 match histograms and crop data
		print("Cropping Image and Setting Metadata...")
		if len(self.cmdInputs['-fi']['variable'][0]) != 4:
			print("Error, enter correct crop points...")
			return

		self.loadFinishedZarr()
		
		self.data['crop_points'] = self.cmdInputs['-fi']['variable'][0]
		self.get_crop_points()
		self.histogram = np.zeros(byteDeapth)
		self.crop_image()
		self.adjust_mean_peak()
		self.zimg = self.finalIMG
		
		self.save_individual_histograms(self.imgPath,self.histogram,0)
		self.data["stage3_flags"] = self.cmdInputs
		self.saveMetadata()

#		#Check if Flythrough and write new flythrough
		if self.cmdInputs['-f']['active'] or self.cmdInputs['-f']['active']:
			print("Creating final flythrough of raw data...")
			self.create_final_flythrough_video_for_data_visualization()
		
		
	def get_crop_points(self):
		self.cropMin_x = self.data['crop_points'][0]
		self.cropMax_x = self.data['crop_points'][1]
		self.cropMin_y = self.data['crop_points'][2]
		self.cropMax_y = self.data['crop_points'][3]
		
		
		if self.cropMax_x != -1:
			self.width = self.cropMax_x - self.cropMin_x
		else:
			self.width = self.zimg.shape[1]
		if self.cropMax_y != -1:
			self.height = self.cropMax_y - self.cropMin_y
		else:
			self.height = self.zimg.shape[2]
		
		index = self.zimg.shape[0]
		self.finalIMG = create_zarr_file(self.imgPath,self.fname,self.width,self.height,index)
	
	
	def loadFinishedZarr(self,fileName = 'rawData'):
		zpath = self.imgPath + f'{fileName}.zarr'
		try:
			self.zimg = da.from_zarr(zpath, component="data/0/")
		except:
			print("Opening Image failed...")
			self.zimg = None

	def crop_image(self):
		self.mean = 0
		index = self.zimg.shape[0]
		for i in tqdm(range(index)):
			image = np.array(self.zimg[i]).astype("uint16")
			image = image[self.cropMin_x:self.cropMax_x,self.cropMin_y:self.cropMax_y]
			self.mean += self.find_max_intensity_value_of_image(image)
			self.finalIMG[i] = image
		self.histogram = self.histogram // index
		self.mean = self.mean // index
	
	def find_smoothed_histogram_signal_of_image(self,image):
		histogram = image_histogram(image)
		try:
			self.histogram += histogram
		except:
			pass
		smooth_signal = savgol_filter(histogram, 50, 2)
		smooth_signal[0:100] = 0
		smooth_signal[4050:-1] = 0
		return smooth_signal
	
	def find_max_intensity_value_of_image(self,image):
		curve = self.find_smoothed_histogram_signal_of_image(image)
		return np.argmax(curve)
	
	def adjust_mean_peak(self):
		index = self.zimg.shape[0]
		for i in tqdm(range(index)):
			self.finalIMG[i] = self.baseline_adjust_image(self.finalIMG[i])
	
	def baseline_adjust_image(self,image):
		mean = self.find_max_intensity_value_of_image(image)
		image = image.astype(float)
		image += self.mean - mean
		image = image.astype('uint16')
		return image
	
	def crop_and_match_hisograms(self):
		index = self.zimg.shape[0]
		matchIMG = np.array(self.finalIMG[0]).astype("uint16")
		histPath = self.metaPath + '/histogram/'
		replace_directory(histPath)
		
		for i in tqdm(range(index)):
			image = np.array(self.finalIMG[i]).astype("uint16")
			image = self.find_histogram_of_image(image,histPath,i)
			self.finalIMG[i] = image
	
	def find_constast_and_mean_numbers(self):
		lowerPeak = []
		upperPeak = []
		index = self.zimg.shape[0]
		for i in tqdm(range(index)):
			image = np.array(self.zimg[i]).astype(float)
			
			
			
			histogram = image_histogram(image)
			
			
			smooth_signal = savgol_filter(histogram, 50, 2)
			smooth_signal[0:100] = 0
			smooth_signal[4050:-1] = 0
			lowerPeak.append(np.argmax(smooth_signal))
			upperPeak.append(np.argmax(smooth_signal[lowerPeak[-1]+200:]))
		lowerPeak = np.array(lowerPeak)
		upperPeak = np.array(upperPeak)
		
		mean = np.mean(lowerPeak)
		contrast = np.mean(upperPeak - lowerPeak)
		
		print(mean,contrast)
	
	
		
	
	def find_histogram_of_image(self,image,histPath,index):
		histogram = image_histogram(image)
		smooth_signal = savgol_filter(histogram, 50, 2)
		smooth_signal[0:100] = 0
		smooth_signal[4050:-1] = 0
		
		darkpeak = np.argmax(smooth_signal)
		peakWidth = calculate_fwhm(smooth_signal[darkpeak-150:darkpeak+150])
		contrasting = 200.0 / float(peakWidth)
		
		image = np.array(image).astype(float)
		image = image - darkpeak
		image = contrasting * image
		image = image + 1000



		histogram = image_histogram(image)
		smooth_signal = savgol_filter(histogram, 50, 2)
		smooth_signal[0:100] = 0
		smooth_signal[4050:-1] = 0
		self.save_individual_histograms(histPath,smooth_signal,index)
		
		return image
		
		
	
	
	
	def process_all_images(self):
		self.loadFinishedZarr(self.fname)
		image = np.array(self.zimg[0])
		self.mean = self.find_max_intensity_value_of_image(image)
		self.width = self.zimg.shape[1]
		self.height = self.zimg.shape[2]
		
		if self.cmdInputs['-w']['active']:
			self.windowMin = self.cmdInputs['-w']['variable'][0]
			self.windowMax = self.cmdInputs['-w']['variable'][1]
			self.windowHalf = (self.windowMax - self.windowMin) // 2
			self.windowMid = self.windowMin + self.windowHalf
			self.windowContrast = byteDeapth // (2 * self.windowHalf)
		
		if self.cmdInputs['-bt']['active']:
			elipse_size = self.cmdInputs['-bt']['variable'][0]
			self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(elipse_size,elipse_size))

		if self.cmdInputs['-d']['active']:
			dilate_size = self.cmdInputs['-d']['variable'][0]
			self.dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE,(dilate_size,dilate_size))
			
		if self.cmdInputs['-e']['active']:
			erode_size = self.cmdInputs['-e']['variable'][0]
			self.erode = cv.getStructuringElement(cv.MORPH_ELLIPSE,(erode_size,erode_size))

		if self.cmdInputs['-ma']['active']:
			elipse_size = self.cmdInputs['-ma']['variable'][1]
			self.add = cv.getStructuringElement(cv.MORPH_ELLIPSE,(elipse_size,elipse_size))
			if self.cmdInputs['-ma']['variable'][0] not in modality_type:
				self.cmdInputs['-ma']['active'] = False
				print(f"Addative Modality, {self.cmdInputs['-ma']['variable'][0]}, not recognized, please correct...")
			else:
				self.add_type = modality_type[self.cmdInputs['-ma']['variable'][0]]

		if self.cmdInputs['-ms']['active']:
			elipse_size = self.cmdInputs['-ms']['variable'][1]
			self.subtract = cv.getStructuringElement(cv.MORPH_ELLIPSE,(elipse_size,elipse_size))
			if self.cmdInputs['-ms']['variable'][0] not in modality_type:
				print(f"Subtractive Modality, {self.cmdInputs['-ms']['variable'][0]}, not recognized, please correct...")
				self.cmdInputs['-ms']['active'] = False
			else:
				self.subtract_type = modality_type[self.cmdInputs['-ms']['variable'][0]]

		
		if self.cmdInputs['-png']['active']:
			replace_directory(self.pngPath)
		
		if self.cmdInputs['-bk']['active']:
			i = self.cmdInputs['-bk']['variable'][0]
			image = self.process_image(self.zimg[i],i)
			cv.imwrite(self.outpath + f"/example_image_{i}.png",image/16)
			return
			
		n = self.zimg.shape[0]
		self.pIMG = create_zarr_file(self.imgPath,self.fname + "_processed",self.width,self.height,n)
		for i in tqdm(range(n)):
			self.pIMG[i] = self.process_image(self.zimg[i],i)

		if self.cmdInputs['-f']['active']:
			print("Creating final flythrough of raw data...")
			self.create_final_flythrough_video_for_data_visualization()
		#RECORD WHAT YOU DID IN STAGE 4 DUMMY
		
		
			
	def process_image(self,image,c):
		image = np.array(image).astype(float)
		
		#Adjust Brightness
		if self.cmdInputs['-br']['active']:
			image += self.cmdInputs['-br']['variable'][0]
			image = np.clip(image,0,byteDeapth)
		
		#Perform Blackhat and Tophat Contrasting
		if self.cmdInputs['-bt']['active']:
			topHat = cv.morphologyEx(image, cv.MORPH_TOPHAT, self.kernel)
			blackHat = cv.morphologyEx(image, cv.MORPH_BLACKHAT, self.kernel)
			image = image + topHat - blackHat
		
		#Perform erosion
		if self.cmdInputs['-e']['active']:
			image = cv.erode(image,self.erode,iterations = 1)
			
		if self.cmdInputs['-d']['active']:
			image = cv.dilate(image,self.dilate,iterations = 1)
		
		#Adds or subtracts an imaging modality:
		if self.cmdInputs['-ma']['active']:
			morph = cv.morphologyEx(image, self.add_type, self.add)
			image = image + morph
			
		if self.cmdInputs['-ms']['active']:
			morph = cv.morphologyEx(image, self.subtract_type, self.subtract)
			image = image - morph

		#Perform Windowing
		if self.cmdInputs['-w']['active']:
			image -= self.windowMid
			image *= self.windowContrast
			image += byteDeapth / 2
			image = np.clip(image,0,byteDeapth)
		
		if self.cmdInputs['-sb']['active']:
			image = overlay_scalebar_on_image(image,16)
		
		if self.cmdInputs['-png']['active']:
			cv.imwrite(self.pngPath + f"/image_{c}.png",image//16)
		image = image.astype('uint16')
		
		if self.cmdInputs['-8b']['active']:
			image = image //16

		return image


#Program Starts Here
CMD = inputs(sys.argv)

if CMD.cmdInputs['-r']['active']:
	flist = findAllDir(CMD.base_path)
	for f in flist:
		print(f)
		DATA = globals(f,CMD.base_path,CMD.cmdInputs)
		print(f,DATA.allArrays)
else:
	DATA = globals(sys.argv[1],CMD.base_path,CMD.cmdInputs)
	print(sys.argv[1],DATA.allArrays)




#Doing too much... split up salvage and compiler
#Salvagers accepts 2 additional input, List of Arrays To Salvage, path to seed image


#Data Compiler needs 3 additional inputs
#Good Arrays
#Bad Arrays
#Ignored Arrays



