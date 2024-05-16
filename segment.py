import methods as ms
import cv2 as cv
from tqdm import tqdm
import numpy as np

#path = '/media/james/T9/REVA/SR005-CL2-4/MUSE_stitched_acq_4.zarr'
path = '/media/james/T9/data/T1-1/MUSE_stitched_acq_1.zarr'

img, attrs = ms.get_image_from_zarr(path)

d = []
size = 5000

for i in tqdm(range(img.shape[0])):
	size = 5000
	if np.sum(img[i]) > 0:
		crop,mask = ms.segment_out_the_nerve(img[i],i)
		
		mask = img[i] * mask
		mask = mask / 16
		
		d.append(crop[2])
		mask = ms.crop_black_border(mask)
		
#		cv.imwrite(f"./output/mask-real_{i}.png",mask)
		
		if mask.shape[0] > mask.shape[1]:
			mask = mask.T
				
		while mask.shape[1] > size:
			size += 2000
			
		
		image = ms.add_smaller_image_to_larger(mask,size)
		image.astype('uint8')
		
		cv.imwrite(f"./output/mask_{i}.png",image)

d = np.array(d)
print(d,np.amax(d))

