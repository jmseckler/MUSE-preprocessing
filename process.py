import methods as ms
import numpy as np
from tqdm import tqdm
import zarr, cv2, os

from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscale


import matplotlib.pyplot as plt

difference_threshhold = 1000
slices_to_skip = 3

sample = 50
image_to_png = 100

errorlog = []

size = 5000


def master_compiler(fname):
	global errorlog
	
	errorlog = []
	
	print(f"Now processing dataset, {fname}")
	data = {}
	zfiles = ms.zarr_image_lister(fname)
	print('Got File List..')
	
	data['mean'], data['std'] = calculate_mean_intensity(zfiles)
	print('Calculated Means and Standard Deviations...')
	
	data['index'] = find_useable_images_and_reports_index(zfiles,data['mean'], data['std'])
	print('Compiled list of usable indices...')
		
	data['crop'] = find_crop_position(data['index'])
	print('Finished Crop images...')
	
	data['runs'] = get_run_numnber_from_filelist(zfiles)
	print('Finished Acquiring Run numbers')
	
	data['attr'] = record_attributes(data['runs'])
	print('Finished recording attributes...')
	
	compile_images_into_single_zarr(data,fname)
	
	path = ms.output_path + '/' + fname
	ms.save_arributes(path,data)

	with open(path + '/error_log.txt', 'w') as f:
		for line in errorlog:
			f.write(f"{line}\n")
	
def calculate_mean_intensity(filelist):
	global errorlog
	counter = 0
	means = []
	for z in filelist:
		try:
			means, counter = load_image_and_get_mean_as_array(z,counter,means)
		except TypeError:#This is where we need to go in and make it revert to tiff stack
			errorlog.append(f"File {z} not found and it was skipped for processing")
	
	means = np.array(means)
	m = np.average(means)
	std = np.std(means)
	
#	x = np.arange(means.shape[0])
#	
#	plt.plot(x,means)
#	plt.show()
	
	return m, std

def load_image_and_get_mean_as_array(z,counter,means):
	img, attrs = ms.get_image_from_zarr(z)
	
	n = len(img)
	for i in range(n):
		if counter % sample == 0:
			mtemp = np.mean(img[i])
			if mtemp > 0:
				means.append(mtemp)
		counter += 1
	return means,counter

def find_useable_images_and_reports_index(filelist,mean,std):
	global errorlog
	
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
					index[counter] = {'file':z,'index':i,'run':get_run_from_index_number(z)}
				elif m > 0:
					pass
                    #Put in code in to track when images are bad as opposed to m == 0
#				elif m == 0:
#					zeros += 1
				counter += 1
		except TypeError:
			errorlog.append(f"File {z} not found and it was skipped for processing")
	
	return index
	

def find_crop_position(index):
	#Finds the minimum crop position and the centroid of the nerve. The result will be a dictionary of indicies with each entry being a list of the order of [x-centroid, y-centroid, radius of image]
	crop = {}
	n = len(index)
	current_zarr_file = ''
	indices_to_delete = []
	
	radius = 2500
	
	for i in tqdm(index):
		if index[i]['file'] != current_zarr_file:
			img, attrs = ms.get_image_from_zarr(index[i]['file'])
			current_zarr_file = index[i]['file']
		
		crop[i], mask = ms.segment_out_the_nerve(img[index[i]['index']])
		mask = ms.crop_black_border(mask)
		
		if mask.shape[0] > mask.shape[1]:
			mask = mask.T
		
		crop[i][2] = mask.shape[1]
		
		if mask.shape[1] > radius:
			radius = mask.shape[1]
	
	crop['radius'] = radius
	return crop

def record_attributes(runs):
	global errorlog
	a = {}
	for n in runs:
		try:
			fname = runs[n]
			img, attrs = ms.get_image_from_zarr(runs[n])
			a[n] = {}
			for attr in attrs.keys():
				a[n][attr] = attrs[attr]
		except AttributeError:
			errorlog.append(f"File {runs[n]} had no attributes and none were records")
	return a

def get_run_numnber_from_filelist(filelist):
	index = {}
	
	for z in filelist:
		run = get_run_from_index_number(z)
		index[run] = z
	return index

def get_run_from_index_number(z):
	run = z.split('.')[-2]
	return int(run.split('_')[-1])
	
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
	
def compile_images_into_single_zarr(data,fname):
	global errorlog
	index = data['index']
	count = len(index)

	while data['crop']['radius'] > size:
		size += 1000
	
	path = ms.output_path + '/' + fname
	png_path = path + '/png/'
	fail_path = path + '/fail/'
	
	ms.replace_directory(path)
	ms.replace_directory(png_path)
	ms.replace_directory(fail_path)

	#Create Folder Structure
	chunk = (4, size,size)
	
	path = ms.output_path + '/' + fname
	png_path = path + '/png/'
	fail_path = path + '/fail/'
	
	ms.replace_directory(path)
	ms.replace_directory(png_path)
	ms.replace_directory(fail_path)
	
	store = zarr.DirectoryStore(path + '/' + fname + '_temp.zarr', dimension_separator='/')
	root = zarr.group(store=store, overwrite=True)
	muse = root.create_group('muse')

	full_data_size = size
	scaled_down_5x = int(full_data_size / 5)
	scaled_down_10x = int(full_data_size / 10)
	
	compiled = muse.zeros('data', shape=(count, full_data_size, full_data_size), chunks=(4, full_data_size, full_data_size), dtype="i2" )
	downsampled5 = muse.zeros('downscaled_5', shape=(count, scaled_down_5x, scaled_down_5x), chunks=(4, scaled_down_5x, scaled_down_5x), dtype="i2" )
	downsampled10 = muse.zeros('downscaled_10', shape=(count, scaled_down_10x, scaled_down_10x), chunks=(4, scaled_down_10x, scaled_down_10x), dtype="i2" )
	
	indices_to_segment = get_indices_which_will_be_manually_segmented(index)	
	number_of_segments = len(indices_to_segment)
	
	if len(indices_to_segment) > 4:
		segmented = muse.zeros('segment', shape=(number_of_segments, full_data_size, full_data_size), chunks=(4, full_data_size, full_data_size), dtype="i2" )
	else:
		segmented = muse.zeros('segment', shape=(number_of_segments, full_data_size, full_data_size), chunks=(number_of_segments, full_data_size, full_data_size), dtype="i2" )
	
	counter = 0
	segment_counter = 0
	current_zarr_file = ''

	for i in tqdm(index):
		if index[i]['file'] != current_zarr_file:
			img, attrs = ms.get_image_from_zarr(index[i]['file'])
			current_zarr_file = index[i]['file']
		if counter > 0:
			image = ms.process_image(img[i],data['mean'],full_data_size,image)
		else:
			image = ms.process_image(img[i],data['mean'],full_data_size)
		
		compiled[counter] = image
		downsampled5[counter] = cv2.resize(image,dsize = (scaled_down_5x,scaled_down_5x),interpolation=cv2.INTER_CUBIC)
		downsampled10[counter] = cv2.resize(image,dsize = (scaled_down_10x,scaled_down_10x),interpolation=cv2.INTER_CUBIC)
		
		if i in indices_to_segment:
			segmented[segment_counter] = image
			
			ms.save_image(str(i),png_path,image / 16)
			
			segment_counter += 1
			
		counter += 1
	
	p = [compiled, downsampled5, downsampled10]
	multiscales, ct = ms.build_coordinate_transforms_metadata_transformation_from_pixel()
	
	store_real = parse_url(path + '/' + fname + '.zarr', mode="w").store
	root_real = zarr.group(store_real)
	
	write_multiscale(pyramid=p, group=root_real,  axes=["z", "y", "x"], coordinate_transformations = ct)
	
	#manually fix multiscales metadata to inlude units for axes
	root.attrs["multiscales"] = multiscales
	
	os.remove(path + '/' + fname + '_temp.zarr')
	
	


flist = ms.find_unprocessed_data_folders()
print(flist)
for fname in flist:
	master_compiler(fname)


