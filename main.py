import warnings
warnings.filterwarnings("ignore")

from lib import stage1 as s1
from lib import stage2 as s2
from lib import stage3 as s3
from lib import image_creator as ic
from lib import image_viewer as img
from lib import input_parser as inp
from lib import methods as ms

import sys, os, json, glob
import numpy as np
import dask.array as da

from datetime import datetime


#Variables
dataConversionTags = ["means"]

stages = ['survey','compile','completed']


def convertDataTagToArray(data):
	for tag in dataConversionTags:
		if tag in data:
			try:
				for i in data[tag]:
					data[tag][i] = np.array(data[tag][i])
			except:
				break
	return data

def get_time():
	now = datetime.now()
	date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
	return date_time_str

def load_metadata(path):
	data = {}
	if os.path.exists(path):
		with open(path) as user_file:
			file_contents = user_file.read()
		data = json.loads(file_contents)
		data = convertDataTagToArray(data)
	return data

def path_validator(path):
	path += os.path.sep
	dataPath = path + 'data.dat'
	acqPath = path + 'MUSE_acq_1' + os.path.sep
	musePath = path + 'compiled.zarr'
	
	if os.path.isdir(musePath):
		return 3
	elif os.path.exists(dataPath):
		return 2
	elif os.path.isdir(acqPath):
		return 1
	else:
		return 0


class dataProcessor:
	def __init__(self,cmd,path,outpath):
		CMD = inp.inputs(cmd,path)
		self.cmdInputs = CMD.cmdInputs
		self.compile = CMD.compile
		self.post = CMD.post
		self.path = path
		self.outpath = outpath
		self.import_variables(CMD)
		self.data_state_determine() # Determines which type of data is being loaded and sets everything up
		self.loadMetaData()
		self.findAllValidAcq()
		self.check_output_directory_structure()
		
		self.save_stage_info_in_data()
		print(f"Running file {self.fname} in stage {self.state}")
		match self.state:
			case 0: return
			case 1:
				if self.only_compile: return
				print("Surveying single data file...")
				survey = s1.survey(self.data)
				survey.run(self.surveyPath)
				if ms.stopping: return
				self.data = survey.data
				self.write_survey_file()

			case 2:
				print("Compiling data file...")
				compile_data = s2.compileData(self.data,self.compile)
				compile_data.run()
				self.data = compile_data.data
				self.write_process_file()
			case 3:
				if self.only_compile: return
				print("Postprocessing data...")
				post_data = s3.processData(self.data,self.post)
				post_data.run()
		self.finish_stage_info_in_data()

	def check_output_directory_structure(self):
		pngFolder = 'png' + os.path.sep
		surveyFolder = 'survey' + os.path.sep
		movieFolder =  'movies' + os.path.sep
		
		if self.state > 1:
			self.outpath = self.path
		elif self.state == 1:
			self.data['acquire_path'] = self.path
		
		if 'acquire_path' not in self.data and self.state != 0:
			print("MUSE Acquire path not saved correctly in metadata, this may cause error...")
		elif self.state != 0:
			self.acqPath = self.data['acquire_path']
			acqPath = self.acqPath + 'MUSE_acq_1' + os.path.sep
			if not os.path.isdir(acqPath):
				print(f"MUSE Acquire data path, {self.acqPath}, is no longer valid, this may cause issues")
		
		if 'filename' in self.data:
			self.fname = self.data['filename']
		else:
			self.fname = self.path.split(os.path.sep)[-2]
			self.data['filename'] = self.fname
		
		if self.outpath == inp.cmdInputs['-o']['variable'][0]:
			self.outpath += self.fname + os.path.sep
		
		self.surveyPath = self.outpath + surveyFolder
		self.moviePath = self.outpath + movieFolder
		#Sets up directory structure and wipes old data
		ms.make_directory(self.outpath)
		ms.make_directory(self.surveyPath)
		ms.make_directory(self.moviePath)
		
		self.data["outpath"] = self.outpath
		self.data["survey_path"] = self.surveyPath
		self.data["movie_path"] = self.moviePath
		
		
		if self.png:
			self.pngPath = self.outpath + pngFolder
			ms.make_directory(self.pngPath)
		
		if self.state == 1:
			self.saveMetaData()

	def convertDataTagToArray(self):
		for tag in dataConversionTags:
			if tag in self.data:
				try:
					for i in self.data[tag]:
						self.data[tag][i] = np.array(self.data[tag][i])
				except:
					break

	def convertDataTagTolist(self):
		data = {}
		for key in self.data:
			data[key] = ms.copy(self.data[key])
		
		for tag in dataConversionTags:
			if tag in self.data:
				for i in self.data[tag]:
					try:
						data[tag][i] = self.data[tag][i].tolist()
					except:
						break
		return data

	def data_state_determine(self):
		self.data_set_initial_state()
		self.data_validate_state_choice()
		

	def data_set_initial_state(self):
		self.state = path_validator(self.path)
		if self.state == 0:	
			print(f"The path {self.path} is not valid data, please check this path")
	
	def data_validate_state_choice(self):
		if self.state == 3:
			musePath = self.path + 'compiled.zarr'
			data = da.from_zarr(musePath, component="data/0/")
			if data is None:
				self.state = 2
				print(f"The final data for {self.path} is corrupted, rerunning stage 2...")
			elif not self.post['success']:
				print(f"The post processing survey is not correctly filled out...")
				quit()
		
		if self.compile_data and self.state > 2:
			print("Rerunning compiling stage.")
			self.state = 2
			return
		if self.survey and self.state > 1:
			print("Rerunning surveying stage.")
			self.state = 1
			return
		if self.state == 1 and not self.survey:
			dataPath = self.outpath + 'data.dat'
			if os.path.isdir(self.outpath) and os.path.exists(dataPath):
				self.state = 0
				print(f"The MUSE Acquire Run {self.path} is already run in stage 1 and will be skipped...")
				return
		if self.state == 2 and not self.compile["success"]:
			self.state = 0
			return

	def findAllValidAcq(self):
		if 'runs' in self.data and len(self.data['runs']) > 0:
			self.allArrays = self.data['runs']
			return
		flist = glob.glob(self.path + "*.zarr")
		allRuns = []
		for fname in flist:
			run = fname.split('.')[0]
			run = run.split('_')[-1]
			allRuns.append(run)
		
		self.allArrays = []
		
		for zarrNumber in allRuns:
			if self.loadRunFile(zarrNumber):
				self.allArrays.append(int(zarrNumber))
		self.allArrays = sorted(self.allArrays)
		#Added to prevent error when more than 10 acq are in
		for i in range(len(self.allArrays)):
			self.allArrays[i] = str(self.allArrays[i])
		
		self.data['runs'] = self.allArrays

	def finish_stage_info_in_data(self):
		STAGE = stages[self.state - 1]
		self.data['stages'][STAGE]['end'] = get_time()
		self.data['stages']['current'] = self.state + 1
		self.saveMetaData()
	
	def import_variables(self,CMD):
		for tag in CMD.cmdInputs:
			if tag != '-o':
				setattr(self, CMD.cmdInputs[tag]["base"], CMD.cmdInputs[tag]["active"])
				n = len(CMD.cmdInputs[tag]["names"])
				for i in range(n):
					setattr(self, CMD.cmdInputs[tag]["names"][i], CMD.cmdInputs[tag]["variable"][i])
		if not self.outpath.endswith(os.path.sep):
			self.outpath += os.path.sep
		if not self.outpath.startswith(os.path.sep) and not self.outpath.startswith('.'):
			self.outpath = os.path.sep + self.outpath

	def loadMetaData(self):
		dataPath = self.path + 'data.dat'
		self.data = load_metadata(dataPath)

	def loadRunFile(self,zarrNumber):
		zpath = self.path + 'MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
		self.IMG = img.img(zpath,True)
		if self.IMG.failed:
			print(f"Error zarr file {zpath} is corrupted, please check...")
			return False
		return True

	def saveMetaData(self):
		dataPath = self.outpath + 'data.dat'
		
		readyData = self.convertDataTagTolist()
	
		for key, value in readyData.items():
			if isinstance(value, np.ndarray):
				readyData[key] = value.tolist()
		if 'mean' in readyData:
			readyData['mean'] = int(readyData['mean'])
	
		with open(dataPath, 'w') as f:
			json.dump(readyData, f)

	def save_stage_info_in_data(self):
		if self.state > 3: return
		if 'stages' not in self.data:
			self.data['stages'] = {'current':self.state}
		
		STAGE = stages[self.state - 1]
		
		self.data['stages']['current'] = self.state
		self.data['stages'][STAGE] = {
			'start':get_time(),
			'flags':{}
			}
		for cmd in self.cmdInputs:
			if self.cmdInputs[cmd]['active']:
				self.data['stages'][STAGE]['flags'][cmd] = self.cmdInputs[cmd]
	
	def write_survey_file(self):
		survey_file = open(self.outpath + "survey_form.csv","w")
		survey_file.write("Description,,,,\n")
		survey_file.write("Stain,,,,\n")
		survey_file.write("Counterstain,,,,\n")
		survey_file.write("Upload,No,,,\n")
		
		survey_file.write(",Min,Max,,\n")
		survey_file.write(f"Crop Height,0,{self.data['height_survey']},,\n")
		survey_file.write(f"Crop Width,0,{self.data['width_survey']},,\n")
		survey_file.write(",,,,\n")
		survey_file.write("Rate all Runs: 0 = Skip, 1 = Use, 2 = Ignore,,,,\n")
		survey_file.write("Run #,Type,Shift Height,Shift Width,Final,\n")
		
		for zarrNumber in self.data['shifts']:
			shift_h = self.data['shifts'][zarrNumber][1]
			shift_w = self.data['shifts'][zarrNumber][0]
			survey_file.write(f"Run_{zarrNumber},,{shift_h},{shift_w},{self.data['length'][zarrNumber]},\n")
		survey_file.close()

	def write_process_file(self):
		compile_file = open(self.outpath + "post_process_form.csv","w")
		compile_file.write(f"Description,{self.data['description'][0]},{self.data['description'][1]},{self.data['description'][2]},\n")
		compile_file.write(f"Stain,{self.data['stains'][0]},{self.data['stains'][1]},{self.data['stains'][2]},\n")
		compile_file.write(f"Counterstain,{self.data['counterstains'][0]},{self.data['counterstains'][1]},{self.data['counterstains'][2]},\n")
		
		compile_file.write(",Min,Max,,\n")
		compile_file.write(f"Crop Height,0,{self.data['width_compile']},,\n")
		compile_file.write(f"Crop Width,0,{self.data['height_compile']},,\n")
		compile_file.write(",,,,\n")
		compile_file.write(",Min,Max,Order 0 before steps,1 after steps,\n")
		compile_file.write(f"Windowing,0,4095,0,\n")
		compile_file.write(f"Indicies,0,{self.data['length_compile']},,\n")
		compile_file.write(",,,,\n")
		compile_file.write("Key: 0 = 12-bit Zarr, '1 = 8-bit Zarr, '2 = 8-bit PNG,'3 = 12-bit Zarr/8-bit PNG,\n")
		compile_file.write("Output,3,,,\n")
		compile_file.write(",,,,\n")
		compile_file.write(f"Sample Index,{self.data['length_compile']//2},Put -1 to process all images,otherwise will only wirte single PNG,\n")
		compile_file.write(",,,,\n")
		compile_file.write(f"Flythrough,1,Put 1 for yes and 0 for no,\n")
		compile_file.write(",,,,\n")
		compile_file.write("Step Key,'0 = Dilation,'1 = Erosion,'2 = Opening,'3 = Closing,\n")
		compile_file.write(",'4 = Gradient,'5 = Tophat,'6 = Blackhat,'7 = Blacktop Contrasting,\n")
		compile_file.write(",,,,\n")
		compile_file.write("Step Type,'0 = Replace Image,'1 = Add to Image,'2 = Subtract from Image,\n")
		compile_file.write(",,,,\n")
		compile_file.write("Step,Type,Kernel Size,\n")
		compile_file.close()


class run_finder:
	def __init__(self,cmd):
		CMD = inp.inputs(cmd)
		self.cmd = CMD.cmdInputs
		self.path = CMD.path
		self.get_filelists()
#		self.remove_redunant_paths()
#		self.format_paths()
	
	def format_paths(self):
		n = len(self.pro_path)
		for i in range(n):
			self.pro_path[i] = self.pro_path[i].removesuffix("data.dat")
		n = len(self.acq_path)
		for i in range(n):
			self.acq_path[i] = self.acq_path[i].removesuffix("MUSE_acq_1")

	def get_filelists(self):

		if path_validator(self.path) > 0:
			flist = [self.path]
		else:
			flist = self.file_list_compiler(self.path)
		
		
#		flist = glob.glob(os.path.join(self.path, '**', '*'), recursive=True)
		self.acq_path = []
		self.pro_path = []
		
		for fpath in flist:
			state = path_validator(fpath)
			if state == 1:
				self.acq_path.append(fpath + os.path.sep)
			elif state > 1:
				self.pro_path.append(fpath + os.path.sep)
	
	def file_list_compiler(self,path):
		flist = []
		cpath = path + os.path.sep + '*'
		rawlist = glob.glob(cpath)
		checklist = []
		for f in rawlist:
			state = path_validator(f)
			if os.path.isdir(f) and state > 0:
				flist.append(f)
			elif state == 0 and f != path:
				checklist.append(f)
		for f in checklist:
			addlist = self.file_list_compiler(f)
			for a in addlist:
				flist.append(a)
		return flist
	
	def get_paths(self):
		paths = []
		for path in self.pro_path:
			paths.append(path)
		for path in self.acq_path:
			paths.append(path)

		if self.cmd['-nr']['active']:
			return [paths[0]]
		return paths
	
	def get_outpaths(self):
		outpaths = []
		for path in self.pro_path:
			outpaths.append(self.cmd['-o']['variable'][0])
		for path in self.acq_path:
			outpaths.append(self.cmd['-o']['variable'][0] + path.removeprefix(self.path))

		if self.cmd['-nr']['active']:
			return [outpaths[0]]
		return outpaths
	
	def remove_redunant_paths(self):
		toRemove = []
		for path in self.pro_path:
			data = load_metadata(path)
			if 'acquire_path' in data:
				acqPath = data['acquire_path'] + 'MUSE_acq_1'
				if acqPath in self.acq_path:
					self.acq_path.remove(acqPath)
			else:
				toRemove.append(path)
		
		for path in toRemove:
			self.pro_path.remove(path)
			
runs = run_finder(sys.argv)
flist = runs.get_paths()
outpaths = runs.get_outpaths()

n = len(flist)
if n == 0:
	print("No valid data found, please check input path...")
	quit()

for i in range(n):
	CMD = dataProcessor(sys.argv,flist[i],outpaths[i])

