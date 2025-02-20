from tqdm import tqdm
from lib import image_viewer as img
from lib import image_creator as imgc
from lib import methods as ms
from lib import methods as ms
import os

class processData:
	def __init__(self,data,post):
		self.data = data
		self.post = post
		self.failed = not post["success"]
		self.setup_local_variables_from_data()
	
	def cleanup(self):
		if self.zimg is not None:
			spath = self.path + "post_process_form.csv"
			dpath = self.outpath + os.path.sep + "post_process_form.csv"
			ms.copy_file(spath, dpath)
	
	def define_indicies(self):
		self.index = []
		if self.break_point > 0 and self.break_point < self.length:
			self.index.append(self.break_point)
			self.output_type = 2
			return
		
		n = self.index_bounds[1] - self.index_bounds[0]
		
		if n > 0:
			for i in range(n):
				idx = i + self.index_bounds[0]
				self.index.append(idx)
		else:
			print("Index listed do not make sense, only processing first listed index...")
			self.index.append(self.index_bounds[0])
			self.output_type = 2
	
	def failstate(self):
		print(f"Post-processing form not filled out correctly for {self.data['filename']} please complete this and retry...")
	
	def loadZarrFile(self):
		self.IMG = img.img(self.zpath)
		if self.IMG.failed or ms.stopping:
			print(f"Error zarr file {self.zpath} is corrupted, please check...")
			return False
		self.IMG.setup_post_processing(self.steps,self.windowing,self.crop)
		return True

	def run(self):
		if not self.loadZarrFile(): return
		if self.setup_image_creator(): return
		
		count = 0
		for i in tqdm(self.index):
			image = self.IMG.get_image_with_post_processing(i)
			self.record_image(image,count)
			count += 1
		self.cleanup()

			
	def record_image(self,image,index):
		if self.zimg is not None:
			self.zimg.add_image(image,index)
		if self.pimg is not None:
			img = image / 16
			self.pimg.add_image(img,index)
		
	
	def setup_local_variables_from_data(self):
		if self.failed:
			self.failstate()
			return
		
		self.fname = self.data['filename']
		self.path = self.data["outpath"]
		self.zpath = self.path[:-1] + 'compiled.zarr'
		self.moviePath = self.data["movie_path"]
		self.length = self.data['length_compile']

		self.index_bounds = self.post['index']		
		self.windowing = self.post['window']
		self.crop = self.post['crop']
		self.steps = self.post['steps']

		self.width = self.crop[3] - self.crop[2] + 1
		self.height = self.crop[1] - self.crop[0] + 1

		
		self.output_type = self.post['output']
		self.break_point = self.post['break_index']
		
		self.define_indicies()

	def setup_image_creator(self):
		self.pngPath = self.path + 'png' + os.path.sep
		self.outpath = self.path +  "processed.zarr"
		
		self.zimg = None
		self.pimg = None
		
		match self.output_type:
			case 0:
				remove_directory(self.outpath)
				self.zimg = imgc.imageCreator(self.outpath,self.height-1,self.width-1,self.length)
			case 1:
				remove_directory(self.outpath)
				self.zimg = imgc.imageCreator(self.outpath,self.height-1,self.width-1,self.length,"8b")
			case 2:
				ms.replace_directory(self.pngPath)
				self.pimg = imgc.imageCreator(self.pngPath,self.height,self.width,self.length,"png")
			case 3:
				ms.remove_directory(self.outpath)
				self.zimg = imgc.imageCreator(self.outpath,self.height-1,self.width-1,self.length)
				ms.replace_directory(self.pngPath)
				self.pimg = imgc.imageCreator(self.pngPath,self.height,self.width,self.length,"png")
			case _:
				return True
		return False

