#Data Surveyor program version 1.0, August 29th 2024

#Import libraries
import methods as ms
from tqdm import tqdm
from datetime import datetime
import cv2 as cv
import numpy as np
import sys, os, getpass, ast, json, glob
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt



#Define the commands input variable which is used with Input Parser to interpret all flags the user could put in
cmdInputs = {
	'-o':{"name":"Override Output","types":['str'],"names":['path'],"variable":["./output/"],"active":False,"tooltips":"Changes output directory"},	
	'-su':{"name":"Data Survey","types":[],"names":[],"variable":[],"active":False,"tooltips":"Perform Data Survey"},
	'-r':{"name":"Recursive","types":[],"names":[],"variable":[],"active":False,"tooltips":"Finds all folders in the base directory"},
	'-f':{"name":"Flythrough","types":[],"names":[],"variable":[],"active":False,"tooltips":"Creates a video flythrough (Linux Only)"},
	'-i':{"name":"Individual Curves","types":[],"names":[],"variable":[],"active":False,"tooltips":"Saves the invidual curves for means, variance, and difference"},
	}

#Define the directory structure which the program will use. Version 2.1 will stop wiping all previous data to go with efficency
dataFolder = 'data/'
metaFolder = 'metadata/'
oldFolder = metaFolder + 'history/'
tmpFolder = dataFolder + 'tmp/'
surveyFolder = metaFolder + 'survey/'

dataConversionTags = ['means','focus','difference','histogram','shift']



fly_font = cv.FONT_HERSHEY_SIMPLEX
fly_color = (255,255,255)
fly_thickness = 2
fly_font_scale = 1
fly_index_x = -50
fly_index_y = 50
fly_scale = 5


scalebar_index_x = -50
scalebar_index_y = -250
scalebar_length_pixels = 200
scalebar_length = int(scalebar_length_pixels * 0.9)
scalebar_color = (255,255,255)
scalebar_thickness = 5
scalebar_text_width = 118
scalebar_text_offset = (scalebar_length_pixels - scalebar_text_width) // 2


header_thickness = 10
header_font_scale = 3


dataQualtyCheck = {
	"Mean Intensity":"means",
	"Laplacian Variance":"focus",
	"Image Difference":"difference"
	}

def setup():
	global fname
	inputParser()
	
	if cmdInputs['-r']['active']:
		flist = ms.findAllDir(base_path)
		for f in flist:
			print(f)
			fname = f
			process_images()
	else:
		process_images()
def process_images():
	variableEncode()
	check_output_directory_structure_and_load_all_metadata()
	validate_data_structure_and_metadata()
	if cmdInputs['-f']['active']:
		create_flythrough_video_for_data_visualization()
	if cmdInputs['-i']['active']:
		save_inidividual_curves()
	

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
	print("Command: python post_processing.py <File Name> <Path to Data> <Options>")
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
	global allArray, data
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
	if 'height' not in data or 'width' not in data:
		passer = determine_pixel_dimensions_for_all_images()
	if 'log' not in data:
		data['log'] = ms.logFileLoader(inPath)
		data = data_saver_to_json(data)
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
	data['width'] = {}
	data['height'] = {}
	for zarrNumber in tqdm(allArray):
		zpath = inPath + 'MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
		img, attrs = ms.get_image_from_zarr(zpath)
		if img is None:
			continue
		data['means'][zarrNumber] = np.zeros(img.shape[0])
		data['width'][zarrNumber] = img.shape[1]
		data['height'][zarrNumber] = img.shape[2]
		
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
		if img is None:
			continue		
		data['focus'][zarrNumber] = np.zeros(img.shape[0])
		
		width, height = img[0].shape
		mid_w = width // 2
		mid_h = height // 2
		start_w = mid_w - 500
		end_w = mid_w + 500
		start_h = mid_h - 500
		end_h = mid_h + 500
		
		for i in range(img.shape[0]):
			image = np.array(img[i][start_w:end_w,start_h:end_h])
			
			vimage = image - np.mean(image)
			vimage = 3.0 * vimage
			vimage = vimage + 2027
			vimage = np.clip(vimage,0,4095)
	
			blurred_image = cv.GaussianBlur(vimage, (15,15), 0)
			variance = cv.Laplacian(blurred_image, cv.CV_64F)
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
		if img is None:
			continue		
		data['difference'][zarrNumber] = np.zeros(img.shape[0])
		
		width, height = img[0].shape

		mid_w = width // 2
		mid_h = height // 2
		
		start_w = mid_w - 500
		end_w = mid_w + 500
		start_h = mid_h - 500
		end_h = mid_h + 500
		
		for i in range(img.shape[0]):
			pImage = np.array(img[i-1][start_w:end_w,start_h:end_h])
			image = np.array(img[i][start_w:end_w,start_h:end_h])
			similarity_score, _ = ssim(image, pImage, full=True)
			data['difference'][zarrNumber][i] = similarity_score
			if data['means'][zarrNumber][i] == 0:
				break
	data = data_saver_to_json(data)
	return True


def determine_pixel_dimensions_for_all_images():
	global data
	data['width'] = {}
	data['height'] = {}
	for zarrNumber in allArray:
		zpath = inPath + 'MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
		img, attrs = ms.get_image_from_zarr(zpath)
		if img is None:
			continue
		data['width'][zarrNumber] = img.shape[1]
		data['height'][zarrNumber] = img.shape[2]
	data = data_saver_to_json(data)
	return True
	


def calculate_histogram_for_all_data():
	print('Calculating Histograms for all images')
	global data
	data['histogram'] = {}
	for zarrNumber in tqdm(allArray):
		zpath = inPath + 'MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
		img, attrs = ms.get_image_from_zarr(zpath)
		if img is None:
			continue
		data['histogram'][zarrNumber] = []
		
		for i in range(img.shape[0]):
			data['histogram'][zarrNumber].append(ms.image_histogram(img[i]))
			if data['means'][zarrNumber][i] == 0:
				break
		data['histogram'][zarrNumber] = np.array(data['histogram'][zarrNumber])
	
	data = data_saver_to_json(data)
	return True


def determine_number_of_flythroughs_to_make():
	global data
	
	flythrough_number = 0
	flyThroughRuns = {}
	width = 0
	height = 0
	flyThroughRuns[flythrough_number] = []
	
	for zarrNumber in allArray:
		if len(flyThroughRuns[flythrough_number]) == 0 or (width == data['width'][zarrNumber] and height == data['height'][zarrNumber]):
			flyThroughRuns[flythrough_number].append(zarrNumber)
		else:
			flythrough_number += 1
			flyThroughRuns[flythrough_number] = []
			flyThroughRuns[flythrough_number].append(zarrNumber)
		width = data['width'][zarrNumber]
		height = data['height'][zarrNumber]
	return flyThroughRuns

def create_flythrough_video_for_data_visualization():
	global data, tmpPath
	tmpPath = surveyPath + '/tmp/'
	flythroughs = determine_number_of_flythroughs_to_make()
	
	for fly in tqdm(flythroughs):
		FlyArrays = flythroughs[fly]
		ms.replace_directory(tmpPath)
		counter = 0
		for zarrNumber in FlyArrays:
			zpath = inPath + 'MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
			img, attrs = ms.get_image_from_zarr(zpath)
			if img is None:
				continue
			for i in range(data['length'][zarrNumber]+1):
				image = np.array(img[i])
				image = image / 16
				
				resolution = (int(image.shape[1] / fly_scale), int(image.shape[0] / fly_scale))
				image = cv.resize(image, resolution, interpolation= cv.INTER_LINEAR)
				image = overlay_acu_and_index_number_on_image(image,zarrNumber,i)
				image = overlay_scalebar_on_image(image)
				image = overlay_header_on_image(image)
				
				c = format_image_number_to_10000(counter)
				
				cv.imwrite(tmpPath + f"image_{c}.png",image)
				counter += 1
		compile_pngs_to_movie(fly)
	ms.remove_directory(tmpPath)

def overlay_acu_and_index_number_on_image(image,zarrNumber,i):
	position = create_text_position(image.shape)
	text = f"Acq#{zarrNumber}, Index#{i}"
	cv.putText(image, text, position, fly_font, fly_font_scale, fly_color, fly_thickness, cv.LINE_AA)
	return image


def overlay_scalebar_on_image(image):
	position = create_text_position(image.shape,False)
	
	start_point = position
	end_point = (position[0] + scalebar_length_pixels, position[1])
	cv.line(image, start_point, end_point, scalebar_color, scalebar_thickness)
	
	
	text = f"{scalebar_length} um"
	text_size = cv.getTextSize(text, fly_font, fly_font_scale, fly_thickness)[0]
	scalebar_text_offset = (scalebar_length_pixels - text_size[0]) // 2
	
	text_position = (start_point[0] + scalebar_text_offset, start_point[1] - 10)
	cv.putText(image, text, text_position, fly_font, fly_font_scale, fly_color, fly_thickness, cv.LINE_AA)
	return image


def overlay_header_on_image(image):
	text = fname.capitalize()
	text_size = cv.getTextSize(text, fly_font, header_font_scale, header_thickness)[0]
	
	image_width = image.shape[1]
	text_x = (image_width - text_size[0]) // 2
	text_y = 30 + text_size[1]  # 30 pixels padding from the top
	cv.putText(image, text, (text_x, text_y), fly_font, header_font_scale, fly_color, header_thickness, cv.LINE_AA)
	return image


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


def compile_pngs_to_movie(runNumber):
	cmd = f"ffmpeg -framerate 10 -i {tmpPath}image_%04d.png -c:v libx264 -r 30 -y -pix_fmt yuv420p {surveyPath}flythrough_{runNumber}.mp4"
	stream = os.popen(cmd)
	output = stream.read()
	print(output)


def rewrite_data_survey_file_and_write_survey_images():
	print('Writing Log File and saving a selection of images from all zarr acquisitions')
	path = metaPath + "/quality.csv"
	logFILE = open(path,'w')
	
	now = datetime.now()
	date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
	
	logFILE.write(f"Quality Check Generated on {date_time_str} by {getpass.getuser()}\n")
	
	logFILE.write('Acquistion,# of Images,Width,Height,')
	for tag in dataQualtyCheck:
		logFILE.write(tag + ',,')
	logFILE.write('\n')
	for zarrNumber in allArray:
		logFILE.write(zarrNumber + ',')
		logFILE.write(str(int(data['length'][zarrNumber])) + ',')
		logFILE.write(str(int(data['width'][zarrNumber])) + ',')
		logFILE.write(str(int(data['height'][zarrNumber])) + ',')
		for tag in dataQualtyCheck:
			logFILE.write(prepareDataForSurvey(dataQualtyCheck[tag],zarrNumber) + ',')
		logFILE.write('\n')
		save_averaged_histograms(zarrNumber,data['histogram'][zarrNumber])
		saveSurveyImage(zarrNumber)
	logFILE.close()

	
def prepareDataForSurvey(tag,zarrNumber):
	pData = np.mean(data[tag][zarrNumber][data[tag][zarrNumber] != 0])
	try:
		if tag == 'difference':
			pData = 1000 - 1000 * pData
		pData = int(pData)
	except ValueError:
		pData = 0
	sData = np.std(data[tag][zarrNumber][data[tag][zarrNumber] != 0])
	try:
		if tag == 'difference':
			sData = 1000 - 1000 * sData
		sData = int(sData)
	except ValueError:
		sData = 0
	
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
	if img is None:
		return False
	index = int(data['length'][zarrNumber]) // 2
	image = np.array(img[index])
	cv.imwrite(surveyPath + f"example_{zarrNumber}.png",image/16)
	

def save_inidividual_curves():
	global data
	write_curve_file('difference')
	write_curve_file('means')
	write_curve_file('focus')
	

def write_curve_file(name):
	curvesFile = open(metaPath + name + ".csv", 'w')
	length = 0
	
	
	for zarrNumber in allArray:
		if data[name][zarrNumber].shape[0] > length:
			length = data[name][zarrNumber].shape[0]
		curvesFile.write(f'Acq#{zarrNumber},')
	curvesFile.write('\n')
	
	
	for i in range(length):
		for zarrNumber in allArray:
			if i < data[name][zarrNumber].shape[0]:
				datum = data[name][zarrNumber][i]
				curvesFile.write(f"{datum},")
			else:
				curvesFile.write(f'0,')
		curvesFile.write('\n')
	curvesFile.close()
	
	
	
	
setup()

print("Completed")
