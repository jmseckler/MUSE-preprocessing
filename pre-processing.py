import methods as ms
import sys, os, cv2, zarr

from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscale
import numpy as np
from tqdm import tqdm

inpath = "/media/revadata/Expansion/data/"
outpath = "/media/revadata/working/processed/"
dest_path = "/mnt/smb-share/processed-data/"

customization = {
	"Downsample":False,
	"Scalebar":False,
	"Windowing":False,
	"Contrast":False,
	"Mean":False,
	"Crop":False
	}

parameters = {
	"path":"",
	"output":"",
	"dest":"",
	"Downsample":ms.downsample,
	"Windowing":[ms.contrast_factor,ms.nerve_factor],
	"Contrast":ms.elipse_size,
	"Mean":ms.mean,
	"Crop":ms.crop
	}

errorlog = []

def input_parser():
	global customization, parameters
	
	n = len(sys.argv)
	
	if n < 2:
		error_message("sample",0)
	
	parameters["path"] = inpath + sys.argv[1] + "/"
	parameters["output"] = outpath + sys.argv[1] + "/"
	parameters["dest"] = dest_path + sys.argv[1] + "-review/"
	
	
	if not os.path.isdir(parameters["path"]):
		error_message("sample",0)
	
	for i in range(n):
		tag = sys.argv[i]
		if tag[0] == "-":
			if tag == "-d":
				customization["Downsample"] = True
				parameters["Downsample"] = int_convert(i+1,i)
			elif tag == "-b":
				customization["Scalebar"] = True
			elif tag == "-w":
				customization["Windowing"] = True
				parameters["Downsample"] = int_convert(i+1,i)
				parameters["Downsample"] = int_convert(i+2,i)
			elif tag == "-ct":
				customization["Contrast"] = True
				parameters["Contrast"] = float(int_convert(i+1,i)) / 10
			elif tag == "-m":
				customization["Mean"] = True
				parameters["Mean"] = int_convert(i+1,i)
			elif tag == "-cp":
				customization["Crop"] = True
				parameters["Crop"] = int_convert(i+1,i)
				parameters["Crop"] = int_convert(i+2,i)
				parameters["Crop"] = int_convert(i+3,i)
				parameters["Crop"] = int_convert(i+4,i)

#Attempts to convert the expected input into integers and return them, if it runs into a problem it throws and error and quits
def int_convert(var_id,tag_id):
	try:
		varr = sys.argv[var_id]
		return int(varr)
	except:
		error_message("input",tag_id)

#Gives an error message to the user when they do something wrong, work on this later to tell them how to correct the issue
def error_message(error_type,tag):
	if error_type == "input":
		print(f"Input parameter {sys.argv[tag]} is not correctly formatted")
	elif error_type == "sample":
		print("Specimen name is incorrect")
	quit()

def name_parser(fname):
	return fname.split("/")[-1]

#This is the function which begins pre-processing
def preprocess(path,sample_name):
	global errorlog
	data = {}
	
	zlist = ms.zarr_image_lister(path)
	if customization["Mean"]:
		data['mean'] = parameters["Mean"]
		data['index'] = find_images_and_report_nonzero_index(zlist)
	else:
		data['mean'], data['std'] = determine_means_and_std(zlist)
		data['index'] = find_useable_images_and_reports_index(zlist,data['mean'], data['std'])
	save_path = correct_data_and_output_zarr(data,sample_name)
	return save_path
	
	

def determine_means_and_std(zlist):
	means = []
	for zfile in zlist:
		img, attr = ms.get_image_from_zarr(zfile)
		n = img.shape[0]
		for i in range(n):
			if np.sum(img[i]) > 0:
				means.append(np.average(img[i]))
	
	means = np.array(means)
	m = np.average(means)
	std = np.std(means)
	return m, std
		

def find_useable_images_and_reports_index(filelist,mean,std):
	global errorlog
	print("Finding Usable Indicies...")
	index = {}
	
	counter = 0
	
	threshhold = 4 * std
	
	zeros = 0
	
	for z in filelist:
		try:
			img, attrs = ms.get_image_from_zarr(z)
			for i in tqdm(range(len(img))):
				m = np.mean(img[i])
				if np.abs(mean-m) < threshhold and m > 0:
					index[counter] = {'file':z,'index':i,'run':ms.path_number_parser(z),'shape':img[i].shape}
				elif m > 0:
					pass
				counter += 1
		except TypeError:
			errorlog.append(f"File {z} not found and it was skipped for processing")
	
	return index

def find_images_and_report_nonzero_index(filelist):
	global errorlog
	print("Finding Usable Indicies...")
	index = {}
	
	counter = 0
	
	for z in filelist:
		try:
			img, attrs = ms.get_image_from_zarr(z)
			for i in tqdm(range(len(img))):
				m = np.mean(img[i])
				if m > 0:
					index[counter] = {'file':z,'index':i,'run':ms.path_number_parser(z),'shape':img[i].shape}
				elif m > 0:
					pass
				counter += 1
		except TypeError:
			errorlog.append(f"File {z} not found and it was skipped for processing")
	return index	

def correct_data_and_output_zarr(data,fname):
	global errorlog
	index = data['index']
	path = parameters["output"] + fname + '/'
	
	ms.replace_directory(path)
	
	runs = []
	for idx in index:
		if index[idx]['run'] not in runs:
			runs.append(index[idx]['run'])
	
	runs = sorted(runs)
	print("Writing data to system...")
	for run in runs:
		write_zarr_file(data,path,"MUSE_acq_" + str(run),[run])
		#ms.convert_zarr_to_napari_readable(path,"MUSE_acq_" + str(run))
		
	return path
	


def write_zarr_file(data,path,zname,run):
	index = {}
	counter = 0
	for idx in data['index']:
		if data['index'][idx]['run'] in run:
			index[counter] = data['index'][idx]
			counter += 1
	
	ztmp = ms.create_basic_zarr_file(path,zname)
	
	zshape, zchunk = shape_definer(len(index),index[0]['shape'][0],index[0]['shape'][1],1)
	full = ztmp.zeros('0', shape=zshape, chunks=zchunk, dtype="i2" )
	
	zshape5, zchunk = shape_definer(len(index),index[0]['shape'][0],index[0]['shape'][1],5)
	down5x = ztmp.zeros('1', shape=zshape5, chunks=zchunk, dtype="i2" )

	zshape10, zchunk = shape_definer(len(index),index[0]['shape'][0],index[0]['shape'][1],10)
	down10x = ztmp.zeros('2', shape=zshape10, chunks=zchunk, dtype="i2" )
	
	indices_to_segment = get_indices_which_will_be_manually_segmented(index)
	number_of_segments = len(indices_to_segment)
	
	
	zseg = ms.create_basic_zarr_file(path,zname + '_seg')
	
	zshape, zchunk = shape_definer(number_of_segments,index[0]['shape'][0],index[0]['shape'][1],1)
	segmented = zseg.zeros('0', shape=zshape, chunks=zchunk, dtype="i2" )
	
	zshape, zchunk = shape_definer(number_of_segments,index[0]['shape'][0],index[0]['shape'][1],5)
	segmented5x = zseg.zeros('1', shape=zshape, chunks=zchunk, dtype="i2" )
	
	zshape, zchunk = shape_definer(number_of_segments,index[0]['shape'][0],index[0]['shape'][1],10)
	segmented10x = zseg.zeros('2', shape=zshape, chunks=zchunk, dtype="i2" )
	
	current_zarr_file = ''
	counter = 0
	segment_counter = 0

	for i in tqdm(index):
		if index[i]['file'] != current_zarr_file:
			img, attrs = ms.get_image_from_zarr(index[i]['file'])
			current_zarr_file = index[i]['file']
		
		image = img[index[i]['index']]
		#change this later
		image = process_img(image,data['mean'])
		full[counter] = image
		down5x[counter] = cv2.resize(image,dsize = (zshape5[2],zshape5[1]),interpolation=cv2.INTER_CUBIC)
		down10x[counter] = cv2.resize(image,dsize = (zshape10[2],zshape10[1]),interpolation=cv2.INTER_CUBIC) 

		if i in indices_to_segment:
			segmented[segment_counter] = image
			segmented5x[segment_counter]= cv2.resize(image,dsize = (zshape5[2],zshape5[1]),interpolation=cv2.INTER_CUBIC)
			segmented10x[segment_counter] = cv2.resize(image,dsize = (zshape10[2],zshape10[1]),interpolation=cv2.INTER_CUBIC) 
			segment_counter += 1
		counter += 1
	
	ms.copy_zarr_attr(path,zname)
	ms.copy_zarr_attr(path,zname + "_seg")
	


def shape_definer(n,x,y,scale):
	zshape = (n,int(x / scale),int(y / scale))
	zchunk = (4,int(x / scale),int(y / scale))
	return zshape, zchunk
		

def process_img(image,mean):
	image = image - np.average(image)
	image = image + mean
	image = np.clip(image,0,4095)
	return image
	
def get_indices_which_will_be_manually_segmented(index):
	n = len(index)
	
	spacing = int(n / 100) - 1
	
	indices_to_use = []
	
	index_numbers = []
	
	if spacing > 0:
		for i in range(100):
			indices_to_use.append((spacing +1)* i)
	elif n > 100:
		for i in range(100):
			indices_to_use.append(i)
	else:
		for i in range(n):
			indices_to_use.append(i)
	
	count = 0
	for i in index:
		if count in indices_to_use:
			index_numbers.append(i)
		count += 1
	
	return index_numbers

input_parser()

flist = ms.find_unprocessed_data_folders(parameters["path"],parameters["output"])

for fname in flist:
	sample_name = name_parser(fname)
	print(f"Now processing {sample_name}...")
	src = preprocess(fname,sample_name)
	
	#Copy Data to NAS
	dest = parameters["dest"] + sample_name +"/"
	try:
		ms.copy_directory(src,dest)
	except:
		pass
	



