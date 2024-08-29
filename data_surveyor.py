#Data Surveyor program version 1.0, August 29th 2024

#Import libraries
import methods as ms
from tqdm import tqdm
from datetime import datetime
import cv2 as cv
import numpy as np
import sys, os, getpass, ast, json
import matplotlib.pyplot as plt


#Define the commands input variable which is used with Input Parser to interpret all flags the user could put in
cmdInputs = {
	'-o':{"name":"Override Output","types":['str'],"names":['path'],"variable":["./output/"],"active":False,"tooltips":"Changes output directory"},	
	'-su':{"name":"Data Survey","types":[],"names":[],"variable":[],"active":False,"tooltips":"Perform Data Survey"},
	}

#Define the directory structure which the program will use. Version 2.1 will stop wiping all previous data to go with efficency
dataFolder = 'data/'
metaFolder = 'metadata/'
oldFolder = metaFolder + 'history/'
tmpFolder = dataFolder + 'tmp/'
surveyFolder = metaFolder + 'survey/'

dataConversionTags = ['means','focus','difference','histogram','shift']

dataQualtyCheck = {
	"Mean Intensity":"means",
	"Laplacian Variance":"focus",
	"Image Difference":"difference"
	}

def setup():
	inputParser()
	check_output_directory_structure_and_load_all_metadata()
	validate_data_structure_and_metadata()


def inputParser():
	global cmdInputs, fname, base_path
	n = len(sys.argv)
	
	if n < 3 and '-h' not in sys.argv:
		print("No filename and/or path given...")
		quit()
	
	if '-h' in sys.argv:
		printHelp()
	
	fname = sys.argv[1]
	base_path = sys.argv[2]
	
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
	variableEncode()

def variableEncode():
	global outpath

	if cmdInputs['-o']['active']:
		outpath = cmdInputs['-o']['variable'][0]
	else:
		outpath = './output/'
	
	if not outpath.endswith((fname + '/')):
		outpath = outpath + fname + '/'
	


def printHelp():
	print("This is a help for Seckler Post Processing Software.")
	print("It expects to accept the input from MUSE REVA Preprocessing Software.")
	print("Command: python post_processing.py <File Name> <Path to Data> <Run Array> <Options>")
	print("")
	for entry in cmdInputs:
		print(generateHelpString(entry,cmdInputs[entry]))
	quit()


def check_output_directory_structure_and_load_all_metadata():
	print("Setting up output folder and loading all previously collected metadata...")
	global data, metaPath, oldPath, imgPath, inPath, surveyPath, tmpPath
	
	inPath = base_path + fname + '/'
	metaPath = outpath + metaFolder
	oldPath = outpath + oldFolder
	imgPath = outpath + dataFolder
	tmpPath = outpath + tmpFolder
	surveyPath = outpath + surveyFolder
	
	#Sets up directory structure and wipes old data
	ms.make_directory(outpath)
	ms.make_directory(metaPath)
	ms.make_directory(oldPath)
	ms.make_directory(surveyPath)
	ms.make_directory(tmpPath)
	ms.replace_directory(imgPath)
	
	data = data_loader_from_json()	


def data_loader_from_json():
	dataPath = metaPath + '/data.dat'
	if os.path.exists(dataPath):
		with open(dataPath) as user_file:
			file_contents = user_file.read()
		data = json.loads(file_contents)
		data = convertDataTagToArray(data)
		
	else:
		data = {}
	return data

def data_saver_to_json(datum):
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
	

def validate_data_structure_and_metadata():
	global allArray
	allArray = ms.findAllZarrs(inPath)
	validatate_zarr_path_and_determine_if_runs_are_valid()
	passer = False
	if 'means' not in data:
		passer = calculate_global_means_for_all_images_and_save_attributes()
	if 'focus' not in data:
		passer = calculate_global_focus_for_all_images()
	if 'difference' not in data:
		passer = calculate_global_adjacent_image_difference_for_all_images()
	if 'histogram' not in data:
		passer = calculate_histogram_for_all_data()
	
	if passer or cmdInputs['-su']['active']:
		rewrite_data_survey_file_and_write_survey_images()


def validatate_zarr_path_and_determine_if_runs_are_valid():
	check_directory_structure(inPath)
	
	for zarrNumber in allArray:
		path = inPath + 'MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
		check_directory_structure(path)


def check_directory_structure(path):
	if not os.path.isdir(path):
		print(f"Path {path} is invalid, please correct...")
		quit()


def calculate_global_means_for_all_images_and_save_attributes():
	print('Calculating Global Means for all images')
	global data
	data['means'] = {}
	data['length'] = {}
	data['attr'] = {}
	for zarrNumber in tqdm(allArray):
		zpath = inPath + 'MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
		img, attrs = ms.get_image_from_zarr(zpath)
		data['attr'][zarrNumber] = attrs
		data['means'][zarrNumber] = np.zeros(img.shape[0])
		
		for i in range(img.shape[0]):
			data['means'][zarrNumber][i] = np.mean(img[i])
			data['length'][zarrNumber] = i
			if data['means'][zarrNumber][i] == 0:
				break
	data = data_saver_to_json(data)
	return True


def calculate_global_focus_for_all_images():
	print('Calculating global focus for all images')
	global data
	data['focus'] = {}
	for zarrNumber in tqdm(allArray):
		zpath = inPath + 'MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
		img, attrs = ms.get_image_from_zarr(zpath)
		data['focus'][zarrNumber] = np.zeros(img.shape[0])
		for i in range(img.shape[0]):
			image = np.array(img[i])
			variance = cv.Laplacian(image, cv.CV_64F)
			data['focus'][zarrNumber][i] = variance.var()
			if data['means'][zarrNumber][i] == 0:
				break
	data = data_saver_to_json(data)
	return True

		
def calculate_global_adjacent_image_difference_for_all_images():
	print('Calculating Adjacent Image Differences For All Images')
	global data
	data['difference'] = {}
	for zarrNumber in tqdm(allArray):
		zpath = inPath + 'MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
		img, attrs = ms.get_image_from_zarr(zpath)
		data['difference'][zarrNumber] = np.zeros(img.shape[0])
		pImage = np.array(img[-1])
		
		for i in range(img.shape[0]):
			image = np.array(img[i])
			
			diff = image - pImage
			diff = np.power(diff,2)
			diff = np.abs(diff)
			diff = np.sum(diff)
			
			data['difference'][zarrNumber][i] = np.sqrt(diff)
			if data['means'][zarrNumber][i] == 0:
				break
	data = data_saver_to_json(data)
	return True


def calculate_histogram_for_all_data():
	print('Calculating Histograms for all images')
	global data
	data['histogram'] = {}
	for zarrNumber in tqdm(allArray):
		zpath = inPath + 'MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
		img, attrs = ms.get_image_from_zarr(zpath)
		data['histogram'][zarrNumber] = []
		
		for i in range(img.shape[0]):
			data['histogram'][zarrNumber].append(image_histogram(img[i]))
			if data['means'][zarrNumber][i] == 0:
				break
		data['histogram'][zarrNumber] = np.array(data['histogram'][zarrNumber])
	
	data = data_saver_to_json(data)
	return True

def image_histogram(image):
	image_array = np.array(image)
	flattened_array = image_array.flatten()
	histogram, bin_edges = np.histogram(flattened_array, bins=4096, range=(0, 4096))
	
	return histogram

def rewrite_data_survey_file_and_write_survey_images():
	print('Writing Log File and saving a selection of images from all zarr acquisitions')
	path = metaPath + "/quality.csv"
	logFILE = open(path,'w')
	
	now = datetime.now()
	date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
	
	logFILE.write(f"Quality Check Generated on {date_time_str} by {getpass.getuser()}\n")
	
	logFILE.write('Acquistion,# of Images,')
	for tag in dataQualtyCheck:
		logFILE.write(tag + ',,')
	logFILE.write('\n')
	
	for zarrNumber in allArray:
		logFILE.write(zarrNumber + ',')
		logFILE.write(str(int(data['length'][zarrNumber])) + ',')
		for tag in dataQualtyCheck:
			logFILE.write(prepareDataForSurvey(dataQualtyCheck[tag],zarrNumber) + ',')
		logFILE.write('\n')
		save_averaged_histograms(zarrNumber,data['histogram'][zarrNumber])
		saveSurveyImage(zarrNumber)
	logFILE.close()

	
def prepareDataForSurvey(tag,zarrNumber):
	pData = np.mean(data[tag][zarrNumber][data[tag][zarrNumber] != 0])
	pData = int(pData)
	sData = np.std(data[tag][zarrNumber][data[tag][zarrNumber] != 0])
	sData = int(sData)
	
	STRING_TO_RETURN = str(pData) + ',' + str(sData)
	return STRING_TO_RETURN
	

def save_averaged_histograms(zarrNumber,histogram):
	hist = np.mean(histogram,axis=0)
	hist[0] = 0
	hist[4050:-1] = 0
	x = np.arange(4096)
	plt.figure(figsize=(10, 6))
	plt.plot(x, hist)  # bin_edges has one extra element
	plt.title(f"Histogram of Pixel Intensities for Acq #{zarrNumber}")
	plt.xlabel("Pixel Intensity")
	plt.ylabel("Frequency")
	plt.savefig(surveyPath + f'histogram_{zarrNumber}.png')
	plt.close()
	

def saveSurveyImage(zarrNumber):
	zpath = inPath + 'MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
	img, attrs = ms.get_image_from_zarr(zpath)
	index = int(data['length'][zarrNumber]) // 2
	image = np.array(img[index])
	cv.imwrite(surveyPath + f"example_{zarrNumber}.png",image/16)
	


setup()
print("Completed")
