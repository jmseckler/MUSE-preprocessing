from lib import methods as ms
import os, shutil, zarr, glob
import dask.array as da
import cv2 as cv
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import median_filter
import numpy as np
import tifffile as tiffio


header_thickness = 10
header_font_scale = 6

fly_font = cv.FONT_HERSHEY_SIMPLEX
fly_color = (255,255,255)
fly_thickness = 10
fly_font_scale = 5
fly_index_x = -50
fly_index_y = 50
fly_scale = 5

scalebar_length_pixels = 1000
scalebar_index_x = -50
scalebar_index_y = -50 - scalebar_length_pixels
scalebar_length = int(scalebar_length_pixels * 0.9)
scalebar_color = (255,255,255)
scalebar_thickness = 5
scalebar_text_width = 118
scalebar_text_offset = (scalebar_length_pixels - scalebar_text_width) // 2

maximum_length_of_tiff_to_check = 1000

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

def crop_to_center(img):
	width,height = img.shape
	mid_w = width // 2
	mid_h = height // 2
	start_w = mid_w - ms.shift_crop_size
	end_w = mid_w + ms.shift_crop_size
	start_h = mid_h - ms.shift_crop_size
	end_h = mid_h + ms.shift_crop_size
	return start_w, end_w, start_h, end_h

def determine_tiff_path(zpath):
	zarrNumber = zpath.split('_')[-1]
	zarrNumber = zarrNumber[:-5]
	listPath = zpath.split(os.path.sep)
	listPath.pop()
	
	
	inPath = ''
	for l in listPath:
		inPath += l + os.path.sep
	
	tiffPath = inPath + 'MUSE_acq_' + zarrNumber + os.path.sep
	tlist = glob.glob(tiffPath + '*.tif')
	tlist = sorted(tlist)
	return tlist


def focus(img):
#	if img is None: return 1000
	vimage = img - np.mean(img)
	vimage = 3.0 * vimage
	vimage = vimage + 2027
	vimage = np.clip(vimage,0,4095)
	blurred_image = cv.GaussianBlur(vimage, (15,15), 0)
	variance = cv.Laplacian(blurred_image, cv.CV_64F)
	return variance.var()

def save_single_panel_tiff_as_zarr_file(zpath):
	tlist = determine_tiff_path(zpath)
	if len(tlist) == 0:
		return False 
	
	ms.remove_directory(zpath)

	store = zarr.DirectoryStore(zpath, dimension_separator=os.path.sep)
	root = zarr.group(store=store, overwrite=True)
	data = root.create_group('muse')


	z = 0
	for t in tlist:
		for i in range(maximum_length_of_tiff_to_check):
			try:
				image = tiffio.imread(t, key = i)
			except IndexError:
				break
			z += i

	image = tiffio.imread(t, key = 0)
	
	x, y = image.shape
	
	zshape, zchunk = shape_definer(z,x,y,1)
	full = data.zeros('stitched', shape=zshape, chunks=zchunk, dtype="i2" )

	zcount = 0
	for t in tlist:
		for i in range(maximum_length_of_tiff_to_check):
			try:
				image = tiffio.imread(t, key = i)
				full[zcount] = image
			except IndexError:
				break
			zcount += 1
	return True

def shape_definer(n,x,y,scale):
	zshape = (n,int(x / scale),int(y / scale))
	zchunk = (4,int(x / scale),int(y / scale))
	return zshape, zchunk

def similarity(img1,img2):
	img1 = img1.astype(np.uint16)
	img2 = img2.astype(np.uint16)
	similarity_score, _ = ssim(img1, img2, full=True)
	s = int(10000 * (1 - similarity_score))
	return s


class img():
	def __init__(self,inpath,acq=False):
		self.inPath = inpath
		self.acq = acq
		self.failed = True
		
		if not self.inPath.endswith('.zarr'):
			return
		
		self.loadZarrFile()
		
		if self.acq:
			self.get_run_number_from_path()
		
		if self.zIMG is None:
			return

		self.length = self.zIMG.shape[0]
		self.height = self.zIMG.shape[1]
		self.width = self.zIMG.shape[2]
		
		self.setup_kernels()
		self.crop = None
		self.window = None
		self.steps = None
		self.byteDepth = 4096
		
		self.failed = False
		return
	
	def contrast_enhance_for_image_align(self,image):
		contrast = 2.0
		elipse_size = 30
		kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(elipse_size,elipse_size))
		mean = np.mean(image)
		image.astype('float')
		
		topHat = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)
		blackHat = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)
		contrast_image = image + topHat - blackHat
		
		contrast_image = image - mean
		contrast_image = contrast * contrast_image
		
		contrast_image = contrast_image + mean
		contrast_image = contrast_image.astype('uint8')
		
		return contrast_image
	
	def crop_image(self,image):
		return image[self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]]

	
	def define_kernel(self,key,size):
		self.kernel[key] = cv.getStructuringElement(cv.MORPH_ELLIPSE,(size,size))
	
	def denoise_image(self,img):
		denoised_image = median_filter(img, size=ms.median_filter_zie)
		return denoised_image
	
	def find_image_position(self,large_image, small_image):
		result = cv.matchTemplate(large_image, small_image, cv.TM_CCOEFF_NORMED)
		min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
		
		shifty, shiftx = max_loc
		sizex, sizey = large_image.shape
		
		sizex /= 2
		sizey /= 2
		
		sizex -= ms.shift_crop_size
		sizey -= ms.shift_crop_size
		
		LOC = np.array([shiftx - sizex,shifty - sizey])
		
		if np.abs(LOC[0]) > ms.max_shift_x:
			LOC[0] = 0
		if np.abs(LOC[1]) > ms.max_shift_y:
			LOC[1] = 0
		
		LOC = LOC.astype('int64')
		
		return LOC

	def get_crop_points_to_center(self,small,large):
		diff = (large - small) // 2
		start = diff
		end = large - diff
		return start,end

	def get_image(self,index,width = 0,height = 0,bitRate=12):
		try:
			image = np.array(self.zIMG[index])
		except RuntimeError:
			print(f"Error occurred in image {index} of zarr with path {self.inPath}")
			return None
		
		if bitRate == 8:
			image = image // 16
		
		if (width == 0 and height == 0) or (width == self.width and height == self.height):
			return image
		
		if width < self.width:
			start,end = self.get_crop_points_to_center(width,self.width)
			image = image[start:end]
		elif width > self.width:
			newImage = np.zeros((width,image.shape[2]))
			start,end = self.get_crop_points_to_center(self.width,width)
			newImage[start:end] = image
			image = newImage
		
		if height < self.height:
			start,end = self.get_crop_points_to_center(height,self.height)
			image = image[:,start:end]
		elif height > self.height:
			newImage = np.zeros((image.shape[1],height))
			start,end = self.get_crop_points_to_center(self.height,height)
			newImage[:,start:end] = image
			image = newImage
		return image

	def get_image_from_zarr_as_dask_array(self):
		try:
			zimg = da.from_zarr(self.inPath, component="muse/stitched/")
			return zimg
		except:
			if save_single_panel_tiff_as_zarr_file(self.inPath):
				zimg = da.from_zarr(self.inPath, component="muse/stitched/")
				return zimg
			else:
				print(f"Filename, {self.inPath} is corrupted or incorrect and did not produce a file from zarr file...")
				return None

	def get_image_with_post_processing(self,index):
		if self.window is None or self.steps is None:
			print("Post Processing Criteria are not properly set, please correct...")
			return
		
		image = self.get_image(index)
		image = image.astype('float')
		
		if self.window[2] == 0:
			image = self.window_image(image)

		if self.crop is not None:
			image = self.crop_image(image)
		
		if self.denoise:
			image = self.denoise_image(image)

		image = self.post_processing_step(image)

		if self.window[2] != 0:
			image = self.window_image(image)

		
		image = np.clip(image,0,self.byteDepth-1)
		return image


	def get_image_with_shift(self,index,output_width,output_height,shift,scale = 1.0,gridline = False,title=None,label=False,scalebar=False,reduce_bits = False,crop = None):
		#Index is an int representing the index of the zarr to pull the image from
		#Output_height and output_width are the size of the image that the image will be matted on. These must be larger than the image size
		#Scale is the scale that the image will be resized to
		#gridlnes is a boolean variable which will add lines to the image at 500 pixel intervals before scaling
		#Title adds title text to the image if variable is not None, must be string
		#Label adds the index label to the image if true
		#Scale bar adds a scale bar to the image
		#Reduce Bits downscales the image to 8 bits if needed
		
		if output_width < self.width or output_height < self.height:
			print("Error, image requested is larger than the matt size.")
			print(output_width, self.width, output_height, self.height)
			return
		
		if reduce_bits:
			image = self.get_image(index,bitRate=8)
		else:
			image = self.get_image(index)
		if image is None:
			return None
		
		output_image = np.zeros((output_height,output_width))
		
		start_w = shift[0]
		end_w = shift[0] + self.width
		start_h = shift[1]
		end_h = shift[1] + self.height
		
		if output_height < end_h or output_width < end_w:
			print("Size Failure")
			print("Width",self.width,output_width,end_w)
			print("Height",self.height,output_height,end_h)
			print("Shift",shift[0],shift[1])
			return
		
		output_image[start_h:end_h,start_w:end_w] = image

		if crop is not None:
			self.crop = crop
			output_image = self.crop_image(output_image)
		
		if gridline:
			output_image = self.overlay_grid_lines_on_image(output_image)
		
		if scalebar:
			output_image = self.overlay_scalebar_on_image(output_image,not reduce_bits)
		
		if label:
			output_image = self.overlay_run_and_index_number_on_image(output_image,index)
		
		if title is not None:
			output_image = self.overlay_header_on_image(output_image,title,not reduce_bits)

		if scale != 1.0:
			resolution = (int(output_image.shape[1] / scale), int(output_image.shape[0] / scale))
			output_image = cv.resize(output_image, resolution, interpolation= cv.INTER_LINEAR)
		
		return output_image

	
	def get_shift_between_images(self,ref,img):
		ref = self.contrast_enhance_for_image_align(ref)
		img = self.contrast_enhance_for_image_align(img)
		
		start_w, end_w, start_h, end_h = crop_to_center(img)
		IMG = img[start_w: end_w, start_h: end_h]
		return self.find_image_position(ref,IMG)

	def get_run_number_from_path(self):
		run = self.inPath.split('_')[-1]
		run = run.split('.')[0]
		
		try:
			self.run = int(run)
		except:
			self.run = None

	def loadZarrFile(self):
		if self.acq:
			self.zIMG = self.get_image_from_zarr_as_dask_array()
		else:
			self.zIMG = da.from_zarr(self.inPath, component="data/0/")

	def match_histogram(self,reference,match):
		img2 = self.get_image(match)
		return match_histograms(img2, reference)

	def overlay_grid_lines_on_image(self,image):
		vLines = image.shape[0] // ms.spacing
		hLines = image.shape[1] // ms.spacing
		
		for h in range(hLines):
			cv.line(image, (ms.spacing * (h + 1),ms.spacing), (ms.spacing * (h + 1),(vLines) * ms.spacing), ms.scalebar_color, 10)
		for v in range(vLines):
			cv.line(image, (ms.spacing,ms.spacing * (v + 1)), ((hLines) * ms.spacing,ms.spacing * (v + 1)), ms.scalebar_color, 10)
		return image

	def overlay_header_on_image(self,image,title,upbit = False):
		if upbit:
			brate = 16
		else:
			brate = 1

		text = title.capitalize()
		text_size = cv.getTextSize(text, fly_font, header_font_scale, header_thickness)[0]
		
		flyColor = (brate * fly_color[0],brate * fly_color[1],brate * fly_color[2])
		
		image_width = image.shape[1]
		text_x = (image_width - text_size[0]) // 2
		text_y = 30 + text_size[1]  # 30 pixels padding from the top
		cv.putText(image, text, (text_x, text_y), fly_font, header_font_scale, flyColor, header_thickness, cv.LINE_AA)
		return image


	def overlay_run_and_index_number_on_image(self,image,i):
		position = create_text_position(image.shape)
		if self.acq:
			text = f"Run#{self.run}, Index#{i}"
		else:
			text = f"Index#{i}"
		cv.putText(image, text, position, fly_font, fly_font_scale, fly_color, fly_thickness, cv.LINE_AA)
		return image
	
	def overlay_scalebar_on_image(self,image,upbit=False):
		position = create_text_position(image.shape,False)
		
		if upbit:
			brate = 16
		else:
			brate = 1
		
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
	
	def post_processing_step(self,image):
		tmp = np.zeros(image.shape)
		for step in self.steps:
			key = ms.postprocess_key[step['cmd']]
			if step['kernel'] > 0:
				self.define_kernel(key,step['kernel'])
			
			if step['type'] == 0:
				image = image + tmp
				image = getattr(self,key)(image)
				tmp = np.zeros(image.shape)
			elif step['type'] == 1:
				tmp = tmp + getattr(self,key)(image)
			elif step['type'] == 2:
				tmp = tmp - getattr(self,key)(image)
		image = image + tmp
		return image
		
	
	def post_processing_windowing_set(self):
		if self.window is None: return
		self.windowHalf = (self.window[1] - self.window[0]) // 2
		self.windowMid = self.window[0] + self.windowHalf
		self.windowContrast = float(self.byteDepth) / (2 * float(self.windowHalf))
	
	def setup_kernels(self):
		self.kernel = {}
		for key in ms.postprocess_key:
			self.define_kernel(key,ms.kernel[key])
	
	def setup_post_processing(self,steps,windowing,crop,denoise=True):
		#Index is an int representing the index of the zarr to pull the image from
		#Steps are post-processing steps taken from the input form
		#Windowing is the max and min pixel intensitues, taken from the input form
		#Crop is the height and width of the output image, taken from input form
		self.crop = crop
		self.window = windowing
		self.steps = steps
		self.denoise = denoise
		self.post_processing_windowing_set()

	def window_image(self,image):
		image -= self.windowMid
		image *= self.windowContrast
		image += self.byteDepth // 2
		image = np.clip(image,0,self.byteDepth-1)
		return image


#Post processing functions, I'm keeping them separate because I need to see them all together
	def dilation(self,image):
		kernel = self.kernel['dilation']
		return cv.dilate(image,kernel,iterations = 1)

	def erosion(self,image):
		kernel = self.kernel['erosion']
		return cv.erode(image,kernel,iterations = 1)

	def opening(self,image):
		kernel = self.kernel['opening']
		return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

	def closing(self,image):
		kernel = self.kernel['closing']
		return cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
	
	def gradient(self,image):
		kernel = self.kernel['gradient']
		return cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)

	def tophat(self,image):
		kernel = self.kernel['tophat']
		return cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)

	def blackhat(self,image):
		kernel = self.kernel['blackhat']
		return cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)
	
	def blacktop(self,image):
		kernel = self.kernel['blacktop']
		topHat = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)
		blackHat = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)
		image = image + topHat - blackHat
		return image



class tiff_compiler():
	def __init__(self,inpath,panel_x = 2,panel_y = 2):
		self.inPath = inpath
		self.paths = determine_tiff_path(inpath)
		self.x = panel_x
		self.y = panel_y
		self.panels = panel_x * panel_y
	
	def get_tiff_paths(self):
		flist = glob.glob(path + "*.tif")
		flist = sorted(flist)
		self.paths = []
		self.length = 0
		for f in flist:
			for i in range(maximum_length_of_tiff_to_check):
				try:
					image = tiffio.imread(f, key = i)
				except IndexError:
					break
				self.paths.append([f,i])
				self.length += i

	def get_panel(index):
		if index >= self.length: return
		for t in tiffs:
			if t[1] <= index:
				index -= t[1]
			else:
				image = tiffio.imread(t[0], key = index)
				return image




	def get_image(self,index):
		if index >= self.length // self.panels: return []
		panels = []
		start = index * self.panels
		if index % 2 == 0:
			for i in range(self.panels):
				j = start + i
				panels.append(get_panel(j))
		else:
			for i in range(self.panels):
				j = start + self.panels - i - 1
				panels.append(get_panel(j))
		return panels

	

