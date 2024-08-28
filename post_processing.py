#Postprocessing programe Version 2.0, August 18th, 2024

import methods as ms
import sys, getpass, ast, os, json, multiprocessing
import numpy as np
import cv2 as cv
from tqdm import tqdm

cmdInputs = {
	'-bk':{"name":"Single Image Run","types":['int'],"names":["break point"],"variable":[0],"active":False,"tooltips":"Skips to image listed, analyzes that image, and that stops program. Default 0th image"},
	'-bt':{"name":"Enhanced Contrasting","types":[],"names":[],"variable":[],"active":False,"tooltips":"Preforms enhanced contrast enhancement using TopHat and BlackHat Imaging Modalities"},
	'-cp':{"name":"Crop Image","types":['int','int','int','int'],"names":['height min','height max','width min','width max'],"variable":[0,-1,0,-1],"active":False,"tooltips":"Crops the image to the specified height and width. Default: Will not crop"},
	'-ct':{"name":"Normal Contrasting","types":['float'],"names":['contrast factor'],"variable":[1.0],"active":False,"tooltips":"Contrasts the data according to new_px = factor * (old_px - mean) + 2055. Default: Factor = 1"},
	'-d':{"name":"Downsample Image","types":['int'],"names":['scaling factor'],"variable":[4],"active":False,"tooltips":"Downscale data by whatever factor the user inputs. Default: 4"},
	'-h':{"name":"Help","types":[],"names":[],"variable":[],"active":False,"tooltips":"Generates and prints help message"},
	'-m':{"name":"Mean Locking","types":['float','float'],"names":['mean', 'std'],"variable":[2055.0,10000],"active":False,"tooltips":"Override mean to save time, you must input the mean and the standard deviation as integers of floating point values"},
	'-n':{"name":"Normalize Background","types":['int'],"names":['radius'],"variable":[150],"active":False,"tooltips":"Normalizes the background of a run when the light was misaligned, user must input the radius for normalization convolution. Default: 150 pixels"},
	'-o':{"name":"Override Output","types":['str'],"names":['path'],"variable":["./output/"],"active":False,"tooltips":"Changes output directory"},	
	'-p':{"name":"Processors Used","types":['int'],"names":['processors'],"variable":[4],"active":False,"tooltips":"Sets the maximum number of processing cores that program will used. Default is 4"},
	'-sb':{"name":"Add Scalebar","types":[],"names":[],"variable":[],"active":False,"tooltips":"Adds scalebar to images outputed"},
	'-sk':{"name":"Skip Alignment","types":[],"names":[],"variable":[],"active":False,"tooltips":"Skips aligning between runs"},
	'-su':{"name":"Data Survey","types":[],"names":[],"variable":[],"active":False,"tooltips":"Perform Data Survey"},
	'-v':{"name":"Output video PNGs","types":[],"names":[],"variable":[],"active":False,"tooltips":"Makes a set of downscaled pngs to be compiled into a flythrough in /flythrough/"},
	'-z':{"name":"Output Zarr","types":[],"names":[],"variable":[],"active":False,"tooltips":"Write output to zarr file rather than pngs"}
	}



def setup():
	print("Setting up data structures and validating data...")
	inputParser()
	variableEncode()
	attributes_saver()
	validatate_zarr_path_and_determine_if_runs_are_valid()
	alignBetweenRuns()
	validate_data_file_and_ensure_data_is_correct()
	data_saver_to_json()

	


def inputParser():
	global cmdInputs, fname, base_path, runArray, runLength
	n = len(sys.argv)
	
	if n < 3 and '-h' not in sys.argv:
		print("No filename and/or path given...")
		quit()
	
	if '-h' in sys.argv:
		printHelp()
	
	fname = sys.argv[1]
	base_path = sys.argv[2]
	try:
		runArray = ast.literal_eval(sys.argv[3])
	except:
		runArray = [1]
		print(f"Run input fail, ensure that runs are entered as numbers separated by commas and in brackets, Example: [1,2,4]")
	
	runLength = len(runArray)
	
	for i in range(n):
		tag = sys.argv[i]
		if tag[0] == "-" and tag in cmdInputs:
			cmdInputs[tag]['active'] = True
			
			m = len(cmdInputs[tag]['names'])
			for j in range(m):
				try:
					inputValue = sys.argv[i + j + 1]
					if cmdInputs[tag]['types'][j] == 'float':
						inputValue = float(inputValue)
					elif cmdInputs[tag]['types'][j] == 'str':
						if inputValue[0] == '-':
							cmdInputs[tag]['active'] = False
							print(f"Input {tag} has failed to read in input values, using defaults...")	
					else:
						inputValue = int(inputValue)
					cmdInputs[tag]['variable'][j] = inputValue
				except:
					print(f"Input {tag} has failed to read in input values, using defaults...")
		elif tag == '-jms':
			base_path = '/media/' + getpass.getuser() + '/' + sys.argv[2] + '/data/'



def printHelp():
	print("This is a help for Seckler Post Processing Software.")
	print("It expects to accept the input from MUSE REVA Preprocessing Software.")
	print("Command: python post_processing.py <File Name> <Path to Data> <Run Array> <Options>")
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


def variableEncode():
	global crop_height, crop_width, mean, std, breakPoint, contrastFactor,downScale,bckNormRuns,bckNormPos,outpath, data, pCores, convolutionCircle, kernel
	crop_height = [cmdInputs['-cp']['variable'][0],cmdInputs['-cp']['variable'][1]]
	crop_width = [cmdInputs['-cp']['variable'][2],cmdInputs['-cp']['variable'][3]]
	mean = cmdInputs['-m']['variable'][0]
	std = cmdInputs['-m']['variable'][1]
	breakPoint = cmdInputs['-bk']['variable'][0]
	contrastFactor = cmdInputs['-ct']['variable'][0]
	downScale = cmdInputs['-d']['variable'][0]
	pCores = cmdInputs['-p']['variable'][0]

	if cmdInputs['-o']['active']:
		outpath = cmdInputs['-o']['variable'][0]
	else:
		outpath = './output/'
	
	#Defines convolution circle for background normalization:
	if cmdInputs['-n']['active']:
		diameter = 2 * cmdInputs['-n']['variable'][0] + 1
		center = (cmdInputs['-n']['variable'][0],cmdInputs['-n']['variable'][0])
		convolutionCircle = np.zeros((diameter,diameter))
		cv.circle(convolutionCircle, center, cmdInputs['-n']['variable'][0], 1, 0)
		convolutionCircle = convolutionCircle / np.sum(convolutionCircle)
	
	if cmdInputs['-bt']['active']:
		elipse_size = 30
		kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(elipse_size,elipse_size))

		
	data = data_loader_from_json()


def data_loader_from_json():
	dataPath = outpath + fname + '/data.dat'
	if os.path.exists(dataPath):
		with open(dataPath) as user_file:
			file_contents = user_file.read()
		data = json.loads(file_contents)
		data['means'] = np.array(data['means'])
		for i in data['allMeans']:
			data['allMeans'][i] = np.array(data['allMeans'][i])
		for i in data['shift']:
			data['shift'][i] = np.array(data['shift'][i])
	else:
		data = {}
	return data

def data_saver_to_json():
	dataPath = outpath + fname + '/data.dat'
	
	data['means'] = data['means'].tolist()
	
	for i in data['shift']:
		data['shift'][i] = data['shift'][i].tolist()
	
	for i in data['allMeans']:
		data['allMeans'][i] = data['allMeans'][i].tolist()
	
	with open(dataPath, 'w') as f:
		json.dump(data, f)
	
	data['means'] = np.array(data['means'])
	for i in data['allMeans']:
		data['allMeans'][i] = np.array(data['allMeans'][i])
	for i in data['shift']:
		data['shift'][i] = np.array(data['shift'][i])


def attributes_saver():
	global oldConfig
	oldConfig = attributes_loader()
	ms.replace_directory(outpath + fname + "/")
	
	if cmdInputs['-v']['active']:
		ms.replace_directory(outpath + fname + "/flythrough/")

	if cmdInputs['-su']['active']:
		ms.replace_directory(outpath + fname + "/survey/")

	
	file_path = outpath + fname + "/configuration.txt"
	with open(file_path, 'w') as file:
		file.write(f'Filename: {fname}\n')
		file.write(f'Base Path: {base_path}\n')
		file.write(f'Run Analyzed: {runArray}\n')

		for key, value in cmdInputs.items():
			file.write(f"{value['name']}: {value['active']}, {value['names']}, {value['variable']} \n")

def attributes_loader():
	file_path = outpath + fname + "/configuration.txt"
	if os.path.isfile(file_path):
		with open(file_path, 'r') as file:
			lines = file.readlines()
		config = [line.strip() for line in lines]
	else:
		config = []
	return config

def validatate_zarr_path_and_determine_if_runs_are_valid():
	path = base_path + fname
	check_directory_structure(path)
	
	for zarrNumber in runArray:
		path = base_path + fname + '/MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
		check_directory_structure(path)

def check_directory_structure(path):
	if not os.path.isdir(path):
		print(f"Path {path} is invalid, please correct...")
		quit()

def alignBetweenRuns():	
	global data, runArray, runLength
	passer = determine_if_we_need_to_redo_shift_calculation()
	if not cmdInputs['-sk']['active'] and 'shift' not in data:
		data['shift'] = {}
		
		validRuns = []
		IMG = None
		for i in tqdm(range(runLength)):
			IMG = alignRuns(i,IMG)
			if IMG is not None:
				validRuns.append(runArray[i])
			else:
				runArray = validRuns
				runLength = len(runArray)
				break
		print("Calculated Shifts to coregister between runs")
	elif 'shift' not in data or passer:
		data['shift'] = {}
		for zarrNumber in runArray:
			runNumber = str(zarrNumber)
			data['shift'][runNumber] = np.array([0,0])
		print("Skipping coregistration step, all image shifts are assumed to be 0")
	else:
		print("Old shift values detected, using those values")
	print(data['shift'])


def alignRuns(i,iIMG):
	global data, runArray, runLength
	if i == 0:
		iIMG = get_image_to_align_from_zarr(i)
		data['shift'][str(runArray[i])] = np.array([0,0])
		return iIMG
	fIMG = get_image_to_align_from_zarr(i)
	if iIMG.shape != fIMG.shape:
		print(f"Runs {runArray[i-1]} and {runArray[i]} are not the same shape. Truncating analysis at run number {runArray[i-1]}")
		return None
	else:
		iIMG, data['shift'][str(runArray[i])] = ms.coregister(iIMG,fIMG)
		return iIMG
	

def get_image_to_align_from_zarr(i):
	if i == 0:
		index = -1
	else:
		index = 0
	path = base_path + fname + '/MUSE_stitched_acq_'  + str(runArray[i]) + '.zarr'
	zzarr = ms.get_just_images_from_zarr(path)
	IMG = np.array(zzarr[index])
	IMG = IMG[crop_height[0]:crop_height[1],crop_width[0]:crop_width[1]]
	return IMG


def validate_data_file_and_ensure_data_is_correct():
	global data, mean, std
	
#	if 'means' not in data or 'allMeans' not in data or determine_if_we_need_to_redo_shift_calculation():
#		compute_means_for_all_arrays()
	compute_means_for_all_arrays()
	
	if not cmdInputs['-m']['active']:
		mean = np.mean(data['means'])
		std = np.std(data['means'])
		print(f'Calculated Mean Intensity of all data {mean}, with a standard deviation of {std}')
	if std == 0:
		std = mean
	data['mean'] = mean
	data['std'] = std

	data['offset'] = {}
	offset = 0
	for run in runArray:
		i = str(run)
		data['offset'][i] = offset
		for j in range(data['allMeans'][i].shape[0]):
			if data['allMeans'][i][j] == 0:
				offset += 1
				break
			else:
				offset += 1
		

	
	if 'shape' not in data:
		path = base_path + fname + '/MUSE_stitched_acq_'  + str(runArray[0]) + '.zarr'
		img = ms.get_just_images_from_zarr(path)
		data['shape'] = img[0].shape
	
	for zarrNumber in runArray:
		path = base_path + fname + '/MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
		img = ms.get_just_images_from_zarr(path)
		if data['shape'][0] != img[0].shape[0] and data['shape'][1] != img[0].shape[1]:
			print(f'Run number {zarrNumber} does not have the same shape as {runArray[0]} please process these two runs separately, crop them to the same size, and stitch them together later...')
			quit()


def compute_means_for_all_arrays():
	global data
	data['means'] = []
	
	if 'allMeans' not in data:
		data['allMeans'] = {}
	for zarrNumber in tqdm(runArray):
		if str(zarrNumber) not in data['allMeans']:
			path = base_path + fname + '/MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
			data['allMeans'][str(zarrNumber)] = load_image_and_get_mean_as_array(path,data['means'],str(zarrNumber))
		for m in data['allMeans'][str(zarrNumber)]:
			if m > 0:
				data['means'].append(m)
			else:
				break
		
	data['means'] = np.array(data['means'])


def compute_means_for_all_arrays_():
	global data
	data['means'] = []
	data['allMeans'] = {}
	for zarrNumber in tqdm(runArray):
		path = base_path + fname + '/MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
		data['means'], data['allMeans'][str(zarrNumber)] = load_image_and_get_mean_as_array(path,data['means'],str(zarrNumber))
	data['means'] = np.array(data['means'])


def load_image_and_get_mean_as_array(zpath,means,runNumber):
	global x, y, data
	
	if 'log' not in data:
		data['log'] = {}
	
	img, attrs = ms.get_image_from_zarr(zpath)
	x = img.shape[1]
	y = img.shape[2]
#	print(zpath,img.shape[0])
	data['log'][runNumber] = {'reason':'Completed','total':img.shape[0]}
	
	temp_means = np.zeros(img.shape[0])
	pImage = np.zeros(img[0].shape)
	pImage = pImage[crop_height[0]:crop_height[1],crop_width[0]:crop_width[1]]
	
	for i in range(img.shape[0]):
		if data['shift'][str(runNumber)][0] != 0 or data['shift'][str(runNumber)][1] != 0:
			image = ms.shiftImage(img[i],data['shift'][str(runNumber)])
		else:
			image = img[i]
		
		image = image[crop_height[0]:crop_height[1],crop_width[0]:crop_width[1]]
		image = np.array(image)
		temp_means[i] = np.mean(image)
		
		difference = image - pImage
		difference = np.abs(difference)
		difference = np.sum(difference)
		
		laplacian = cv.Laplacian(image, cv.CV_64F)
		variance = laplacian.var()
		#put in a catch when difference is high and variance is low which says flake is on the block
		data['log'][runNumber]['length'] = i
		if temp_means[i] == 0 or difference <= 1 or variance < 100:
			if temp_means[i] == 0:
				data['log'][runNumber]['reason'] = "Blank Image"
			elif variance < 100:
				data['log'][runNumber]['reason'] = "Blurry Image"
			else:
				data['log'][runNumber]['reason'] = "Repeat Image"
			break
		else:
			pImage = img[i][crop_height[0]:crop_height[1],crop_width[0]:crop_width[1]]
			pImage = np.array(pImage)
		
	return temp_means

def load_image_and_get_mean_as_array_(zpath,means,runNumber):
	global x, y
	img, attrs = ms.get_image_from_zarr(zpath)
	x = img.shape[1]
	y = img.shape[2]
#	print(zpath,img.shape[0])
	
	temp_means = np.zeros(img.shape[0])
	for i in range(img.shape[0]):
		if data['shift'][str(runNumber)][0] != 0 or data['shift'][str(runNumber)][1] != 0:
			image = ms.shiftImage(img[i],data['shift'][str(runNumber)])
		else:
			image = img[i]
		temp_means[i] = np.mean(image[crop_height[0]:crop_height[1],crop_width[0]:crop_width[1]])
		if temp_means[i] == 0:
			break
		
	for m in temp_means:
		if m > 0:
			means.append(m)
		else:
			break
	
	return means,temp_means
	

def determine_if_we_need_to_redo_shift_calculation():
	runs = f"Run Analyzed: {runArray}"
	if runs in oldConfig:
		passer = False
	else:
		passer = True
	return passer


def create_zarr_file():
	global zimg, x, y
	zimg = ms.create_basic_zarr_file(outpath + fname + "/",fname)
	
	z = means.shape[0]
	if crop_image:
		x = crop_height[1] - crop_height[0]
		y = crop_width[1] - crop_width[0]
	else:
		x = data['shape'][0]
		y = data['shape'][1]
	
	zshape, zchunk = ms.shape_definer(z,x,y,1)
	full = zimg.zeros('0', shape=zshape, chunks=zchunk, dtype="i2" )
	return zimg, full


def compileIndices(runNumber):
	maxListSize = 100
	if cmdInputs['-bk']['active']:
		indices = [breakPoint]
	else:
		t = data['allMeans'][str(runNumber)].shape[0]
		n = int(t / maxListSize - 1) + 1
		
		
		indices = []
		count = 0
		
		for i in range(n):
			indices.append([])
			for j in range(maxListSize):
				indices[i].append(count)
				count += 1
				if count >= t:
					break
	return indices

def compileIndices_total(runNumber):
	if cmdInputs['-bk']['active']:
		indices = [breakPoint]
	else:
		t = data['allMeans'][str(runNumber)].shape[0]
		indices = []
		for i in range(t):
			if data['allMeans'][str(runNumber)][i] > 0:
				indices.append(i)
			else:
				break
	return indices


def img_process(index):
	global zfull
	if data['allMeans'][runNumber][index] == 0: return False
	
	image = np.array(rawImage[index])

	if data['shift'][str(runNumber)][0] != 0 or data['shift'][str(runNumber)][1] != 0:
		image = ms.shiftImage(image,data['shift'][runNumber])
	
	m = data['allMeans'][runNumber][index]
	
	#performs basic mean locking and contrasting
	if cmdInputs['-n']['active']:
		meanMap = perform_background_normalization(image)
		image = image - meanMap
		m = np.mean(image)
		image = image - m
	else:
		image = image - m
	image = contrastFactor * image
	image = image + mean
	image = image[crop_height[0]:crop_height[1],crop_width[0]:crop_width[1]]
	
	#performs enhanced contrasting
	if cmdInputs['-bt']['active']:
		topHat = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)
		blackHat = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)
		image = image + topHat - blackHat
	
	image = np.clip(image,0,4095)
	
	if cmdInputs['-sb']['active']:
		image = ms.add_scalebar_to_image(image,4095)

	if cmdInputs['-d']['active']:
		down_points = (int(image.shape[1] / down_scale), int(image.shape[0] / down_scale))
		image = cv.resize(image, down_points, interpolation= cv.INTER_LINEAR)
	
	#Determines counter string
	counter = data['offset'][runNumber] + index
	if counter < 1000:
		if counter < 10:
			c = '000' + str(counter)
		elif counter < 100:
			c = '00' + str(counter)
		else:
			c = '0' + str(counter)
	else:
		c = str(counter)
	
	if cmdInputs['-z']['active']:
		zfull[counter] = image
	elif cmdInputs['-su']['active']:
		cv.imwrite(outpath + fname + f"/survey/image_{c}.png",image/16)
	else:
		cv.imwrite(outpath + fname + f"/image_{c}.png",image/16)

	if cmdInputs['-v']['active']:
		scale = np.amax(image.shape) / base_video_resolution
		if scale > 2:
			scale = 2
		resolution = (int(image.shape[1] / scale), int(image.shape[0] / scale))
		if resolution[0] % 2 != 0:
			resolution = (resolution[0] + 1, resolution[1])
		if resolution[1] % 2 != 0:
			resolution = (resolution[0], resolution[1] + 1)
		video = cv.resize(image, resolution, interpolation= cv.INTER_LINEAR)
		cv.imwrite(outpath + fname + f"/flythrough/image_{c}.png",video/16)
	
	return True


def perform_background_normalization(image):
	convolution = cv.filter2D(src=image, ddepth=-1, kernel=convolutionCircle)
	return convolution


def perform_data_survey():
	save_survey_log_file()
	survey_images()
	data_saver_to_json()
	quit()


def save_survey_log_file():
	path = outpath + fname + "/survey/log.csv"
	logFILE = open(path,'w')
	logFILE.write('Run,Length,Total Length,Reason For Termination\n')
	
	for runNumber in data['log']:
		logFILE.write(str(runNumber) + ',')
		logFILE.write(str(data['log'][runNumber]['length']) + ',')
		logFILE.write(str(data['log'][runNumber]['total']) + ',')
		logFILE.write(str(data['log'][runNumber]['reason']) + '\n')
	logFILE.close()
		
def survey_images():
	global runNumber, rawImage
	indices = []
	for z in data['log']:
		runNumber = str(z)
		path = base_path + fname + '/MUSE_stitched_acq_'  + runNumber + '.zarr'
		rawImage = ms.get_just_images_from_zarr(path)
		index = data['log'][runNumber]['length'] // 2
		img_process(index)
	return indices


setup()

if cmdInputs['-su']['active']:
	perform_data_survey()

if cmdInputs['-z']['active']:
	zimg, zfull = create_zarr_file()


#for zarrNumber in tqdm(runArray):
for zarrNumber in runArray:
	runNumber = str(zarrNumber)
#	print(f"Now procressing run # {zarrNumber}, with {data['offset'][runNumber]} valid images out of a total of {data['allMeans'][runNumber].shape[0]} images")
	path = base_path + fname + '/MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
	rawImage = ms.get_just_images_from_zarr(path)
	
	indicies = compileIndices_total(zarrNumber)
	for  index in tqdm(indicies):
		img_process(index)
	
#	indicies = compileIndices(zarrNumber)
	
#	if cmdInputs['-bk']['active']:
#		results = img_process(indicies[0])
#	else:
#		for indexs in indicies:
#			if len(indexs) < pCores:
#				threadsToStart = len(indexs)
#			else:
#				threadsToStart = pCores
#			if len(indexs) == 1:
#				results = img_process(indexs[0])
#			elif __name__ == "__main__":
#				with multiprocessing.Pool(processes=threadsToStart) as pool:
#					results = pool.map(img_process, indexs)


if cmdInputs['-z']['active']:
	down5x, down10x = finish_making_zarr_file()

data_saver_to_json()

