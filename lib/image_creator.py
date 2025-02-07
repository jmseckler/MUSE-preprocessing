from lib import methods as ms
import numpy as np
import cv2 as cv
import os, zarr

def create_zarr_attr(zpath):
	dst = zpath + '/.zattrs'
	zattr = generate_zattr_file()
	
	zFile = open(dst,'w')
	
	for line in zattr:
		zFile.write(line + '\n')
	zFile.close()


def generate_zattr_file(depth = 12,pixel=0.9):
	x0 = float(pixel)
	y0 = float(pixel)
	x1 = pixel * 5
	y1 = pixel * 5
	x2 = pixel * 10
	y2 = pixel * 10

	zattr = [
		'{',
		'	"multiscales": [',
		'		{',
		'			"axes": [',
		'				{',
		'					"name": "z",',
		'					"type": "space",',
		'					"unit": "micrometer"',
		'				},',
		'				{',
		'					"name": "y",',
		'					"type": "space",',
		'					"unit": "micrometer"',
		'				},',
		'				{',
		'					"name": "x",',
		'					"type": "space",',
		'					"unit": "micrometer"',
		'				}',
		'			],',
		'			"datasets": [',
		'				{',
		'					"coordinateTransformations": [',
		'						{',
		'							"scale": [',
		f'								{depth},',
		f'								{x0},',
		f'								{y0}',
		'							],',
		'							"type": "scale"',
		'						}',
		'					],',
		'					"path": "data/0"',
		'				},',
		'				{',
		'					"coordinateTransformations": [',
		'						{',
		'							"scale": [',
		f'								{depth},',
		f'								{x1},',
		f'								{y1}',
		'							],',
		'							"type": "scale"',
		'						}',
		'					],',
		'					"path": "data/1"',
		'				},',
		'				{',
		'					"coordinateTransformations": [',
		'						{',
		'							"scale": [',
		f'								{depth},',
		f'								{x2},',
		f'								{x2}',
		'							],',
		'							"type": "scale"',
		'						}',
		'					],',
		'					"path": "data/2"',
		'				}',
		'			],',
		'			"name": "/data",',
		'			"version": "0.4"',
		'		}',
		'	]',
		'}',
	]
	return zattr

def shape_definer(n,x,y,scale):
	zshape = (n,int(x / scale),int(y / scale))
	zchunk = (4,int(x / scale),int(y / scale))
	return zshape, zchunk


class imageCreator():
	def __init__(self,PATH,WIDTH,HEIGHT,LENGTH,ITYPE = "12b"):
		self.path = PATH	#Path to the directory that raw data will be stored in, if zarr output, path must be a zarr file, if png this must be a valid directory
		self.width = WIDTH
		self.height = HEIGHT
		self.length = LENGTH
		self.type = ITYPE
		self.failed = False
		
		match self.type:
			case "12b":
				#Prepare to write 12-bit Zarr file
				self.create_zarr_file(self.path,self.width,self.height,self.length)
			case "8b":
				#Prepare to write 8-bit Zarr file
				self.create_zarr_file(self.path,self.width,self.height,self.length)
			case "png":
				#No Preparation needed for png output files
				ms.make_directory(self.path)
				if not os.path.isdir(self.path):
					self.fail_state_activated()
					return
			case _:
				self.fail_state_activated()
				return
				
	def add_image(self,image,index):
		if self.failed: return
		match self.type:
			case "12b":
				#Actually Write Zarr
				self.img[index] = image
			case "8b":
				#Actually Write Zarr
				image = image // 16
				self.img[index] = image
			case "png":
				self.add_image_as_png(image,index)
			case _:
				self.fail_state_activated()
				return
	
	def add_image_as_png(self,image,name):
		if self.failed: return
		cv.imwrite(self.path + f"{name}.png",image)
	
	def create_zarr_file(self,zarr_path,x,y,z):
		if not zarr_path.endswith('.zarr'): return None
	
		ms.remove_directory(zarr_path)
		
		store = zarr.DirectoryStore(zarr_path, dimension_separator='/')
		root = zarr.group(store=store, overwrite=True)
		self.data = root.create_group('data')
		
		zshape, zchunk = shape_definer(z,x,y,1)
		self.img = self.data.zeros('0', shape=zshape, chunks=zchunk, dtype="i2" )
		create_zarr_attr(zarr_path)
		
	def determine_if_image_is_12bit(self,image):
		#Returns True if Image is 12 bit and False if Image is 8 bit
		return np.amax(image) > 255
	
	def fail_state_activated(self):
		self.failed = True
	
	def finish_making_zarr_file(self):
		zshape5, zchunk5 = shape_definer(self.length,self.width,self.height,5)
		down5x = self.data.zeros('1', shape=zshape5, chunks=zchunk5, dtype="i2" )
		zshape10, zchunk10 = shape_definer(self.length,self.width,self.height,10)
		down10x = self.data.zeros('2', shape=zshape10, chunks=zchunk10, dtype="i2" )
		
		for i in range(self.length):
			down5x[i] = cv.resize(self.img[i],dsize = (zshape5[2],zshape5[1]),interpolation=cv.INTER_CUBIC)
			down10x[i] = cv.resize(self.img[i],dsize = (zshape10[2],zshape10[1]),interpolation=cv.INTER_CUBIC) 
		return down5x, down10x
	
	def make_flythrough_movie_from_pngs(self,outpath, mName="raw_flythrough"):
		if not os.path.isdir(outpath) or self.failed: return
		cmd = f'ffmpeg -framerate 10 -i {self.path}image_%04d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -r 30 -y -pix_fmt yuv420p {outpath}{mName}.mp4'
		stream = os.popen(cmd)
		output = stream.read()
		
