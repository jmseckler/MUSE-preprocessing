import methods as ms
import cv2 as cv


from tqdm import tqdm
import numpy as np

path = '/media/james/T9/data/SR005-CL2-4/MUSE_stitched_acq_4.zarr'
#path = '/media/james/T9/data/T1-1/MUSE_stitched_acq_1.zarr'
sample = 50

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

def copy(var):
	return var

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
					cv.imwrite(f"./output/exclude_{i}.png",img[i]/16)
                    #Put in code in to track when images are bad as opposed to m == 0
#				elif m == 0:
#					zeros += 1
				counter += 1
		except TypeError:
			errorlog.append(f"File {z} not found and it was skipped for processing")
	
	return index

def get_run_from_index_number(z):
	run = z.split('.')[-2]
	return int(run.split('_')[-1])


fname = 'SR005-CL2-4'

zfiles = ms.zarr_image_lister(fname)
img, attrs = ms.get_image_from_zarr(path)
mean, std = calculate_mean_intensity(zfiles)

index = find_useable_images_and_reports_index(zfiles,mean,std)

d = []
shift = [0,0]
size = 5000
img_align = img[0]

for i in tqdm(range(img.shape[0])):
	size = 5000
	if np.sum(img[i]) > 0:# and i > 500 and i < 750:
#		cv.imwrite(f"./output/original_{i}.png",img[i]/16)
		
		image = img[i] - np.mean(img[i])
		image = image + mean
		
#		cv.imwrite(f"./output/mean_adjusted_{i}.png",image/16)
		
		
		crop,mask = ms.segment_out_the_nerve(image)
		
#		cv.imwrite(f"./output/mask_{i}.png", 255 * mask)		
		
		mask = image * mask
		mask = mask / 16
		
		d.append(crop[2])
		mask = ms.crop_black_border(mask)
		
#		cv.imwrite(f"./output/mask-real_{i}.png",mask)
		
		
		if mask.shape[0] > mask.shape[1]:
			mask_size = mask.shape[0]
		else:
			mask_size = mask.shape[1]
				
		while mask.shape[1] > size:
			size += 2000
			
		
		image = ms.add_smaller_image_to_larger(mask,size)
		image.astype('uint8')
		
		if i > 0:
			image, s = ms.coregister(img_align,image)
			shift[0] += s[0]
			shift[1] += s[1]
			
			crop[0] += s[0]
			crop[1] += s[1]
						
			img_align = copy(image)
		else:
			img_align = copy(image)
		
		cv.imwrite(f"./output/final_{i}.png",image)

d = np.array(d)
print(d,np.amax(d))

