from tqdm import tqdm
from lib import image_viewer as img
from lib import image_creator as imgc
from lib import methods as ms
import os

class compileData:
	def __init__(self,data,comp):
		self.data = data
		self.compile = comp
		self.failed = not comp["success"]
		self.setup_local_variables_from_data()

	def compile_images_into_single_zarr(self):
		print("Writing compiled zarr images...")
		directory = self.outpath + 'tmp' + os.path.sep
		ms.replace_directory(directory)

		self.length = 0
		
		fmin = self.focus_threshhold
		smin = self.similarity_threshhold_min
		smax = self.similarity_threshhold_max
		
		self.data['images_use_key'] = {}

		for zarrNumber in self.useArray:
			z = self.compile["runs"][zarrNumber]['length']
			self.data['images_use_key'][zarrNumber] = []
			for i in range(z):
				MEAN = self.means[zarrNumber][i]
				FOCUS = self.focus[zarrNumber][i]
				SSIM = self.similarity[zarrNumber][i]
				try:
					pSSIM = self.similarity[zarrNumber][i+1]
				except:
					pSSIM = self.similarity[zarrNumber][i]
					
				if MEAN > 0 and FOCUS > fmin and SSIM > smin and SSIM < smax and pSSIM < smax:
					self.length += 1
					self.data['images_use_key'][zarrNumber].append(1)
				else:
					self.data['images_use_key'][zarrNumber].append(0)
		width = self.compile['crop']['width'][1] - self.compile['crop']['width'][0]
		height = self.compile['crop']['height'][1] - self.compile['crop']['height'][0]
		
		self.zimg = imgc.imageCreator(self.zpath,height,width,self.length)
		self.wIMG = imgc.imageCreator(directory,height,width,self.length,"png")
		
		index = 0
		for zarrNumber in tqdm(self.useArray):
			if not self.loadRunFile(zarrNumber):return
			
			z = self.compile["runs"][zarrNumber]['length']
			for i in range(z):
				MEAN = self.means[zarrNumber][i]
				FOCUS = self.focus[zarrNumber][i]
				SSIM = self.similarity[zarrNumber][i]
				try:
					pSSIM = self.similarity[zarrNumber][i+1]
				except:
					pSSIM = self.similarity[zarrNumber][i]
					
				if MEAN > 0 and FOCUS > fmin and SSIM > smin and SSIM < smax and pSSIM < smax:
					image = self.IMG.get_image_with_shift(i,self.width,self.height,self.shifts[zarrNumber],crop = self.compile['crop']['total'])
					self.zimg.add_image(image,index)
					
					#Write a png for the survey movie
					IMG = self.IMG.get_image_with_shift(i,self.width,self.height,self.shifts[zarrNumber],10.0,False,self.data['filename'],True,True,True,self.compile['crop']['total'])
					c = ms.format_image_number_to_10000(index)
					fname = f"image_{c}"
					self.wIMG.add_image(IMG,fname)
					
					index += 1
		
		self.zimg.finish_making_zarr_file()
	
	def failstate(self):
		print(f"Survey form not filled out for {self.data['filename']} please complete this and retry...")

	def find_focus_and_similarity_of_all_images(self):
		print("Compiling image quality data...")
		self.focus = {}
		self.similarity = {}
		for zarrNumber in tqdm(self.useArray):
			if not self.loadRunFile(zarrNumber): return
			self.focus[zarrNumber] = []
			self.similarity[zarrNumber] = []
			
			for i in range(self.compile["runs"][zarrNumber]['length']):
				image = self.IMG.get_image_with_shift(i,self.width,self.height,self.shifts[zarrNumber],crop = self.compile['crop']['total'])
				if image is None:
					print(f"Run {zarrNumber}, index {i} failed... please check...")
				self.focus[zarrNumber].append(img.focus(image))
				
				if i > 0:
					pImg = self.IMG.get_image_with_shift(i-1,self.width,self.height,self.shifts[zarrNumber],crop = self.compile['crop']['total'])
					self.similarity[zarrNumber].append(img.similarity(image,pImg))
				else:
					self.similarity[zarrNumber].append(0)
	
	def loadRunFile(self,zarrNumber):
		zpath = self.inpath + 'MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
		self.IMG = img.img(zpath,True)
		if self.IMG.failed or ms.stopping:
			print(f"Error zarr file {zpath} is corrupted, please check...")
			return False
		return True

	def record_data(self):
		self.data['similarity_threshhold_min'] = self.similarity_threshhold_min
		self.data['similarity_threshhold_max'] = self.similarity_threshhold_max
		self.data['focus_threshhold'] = self.focus_threshhold
		self.data['focus'] = self.focus
		self.data['similarity'] = self.similarity
		
		for z in self.compile["runs"]:
			self.compile["runs"][z]['shift'] = self.compile["runs"][z]['shift'].tolist()
		self.data['runs'] = self.compile["runs"]

	def setup_local_variables_from_data(self):
		if self.failed:
			self.failstate()
			return
		self.allArrays = self.data['runs']
		self.inpath = self.data['acquire_path']
		self.outpath = self.data["outpath"]
		self.moviePath = self.data["movie_path"]
		self.height = self.data['height_survey']
		self.width = self.data['width_survey']
		self.shifts = self.compile['shifts']
		self.means = self.data['means']
		self.zpath = self.outpath + self.data['filename'] + '.zarr'
		
		self.focus_threshhold = 15
		self.similarity_threshhold_min = 15
		self.similarity_threshhold_max = 100
		
		self.useArray = []
		
		for z in self.allArrays:
			if self.compile["runs"][z]['type'] == 1:
				self.useArray.append(z)
		
	
	def run(self):
		self.stitcher()
		self.find_focus_and_similarity_of_all_images()
		self.compile_images_into_single_zarr()
		self.write_stage_2_movie_pics()
		self.record_data()
	
	def stitcher(self):
		pass

	def write_stage_2_movie_pics(self):
		print("Creating flythrough movie for review...")
		self.wIMG.make_flythrough_movie_from_pngs(self.moviePath,"flythrough")
		ms.remove_directory(self.wIMG.path)

