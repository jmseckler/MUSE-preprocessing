from tqdm import tqdm
from lib import image_viewer as img
from lib import image_creator as imgc
from lib import methods as ms
import numpy as np
import zarr, os, re

logFileName = 'muse_application.log'

report_header = ["Run","Length","Width","Height","Mean Intensity","Shift"]




def logFileLoader(zarrPath):
	path = zarrPath + logFileName
	
	
	panels = {} #Form of Run#: [hPanels, vPanels]
	XYPositions = {} #Form of Run#:[[X1,Y1],[X2,Y2],...]
	ZPositions = {} #Form of Run#:[Z1,Z2,...]
	voxelSize = {} #Form of Run#: (x,y,z) Voxels
	imageSize = {} #form of Run#:[rows,col]
	exposureTime = {} #form of Run#:Time
	runLength = {} #Form of Run#:[Run Length, Final Slice Made, Images Expected, Images Taken]
	trimLength = {} #Form of Trim#:[Run Length,Date Started, Dated Ended]
	dates = {} #Form of Run#:[Start Date/Time, End Date/Time]
	history = [] #Form of ('run',Run#) or ('trim',Trim#)
	
	#Opens the Raw text file form 
	if os.path.isfile(path):
		rawFile = open(path, 'r')
	else:
		return {}
	#regular expressions for capturing information from various row types
	cycletype = re.compile(r"(?P<type>[A-Z]+) CYCLE")
	cyclenum = re.compile(r"CYCLE (?P<cycle>\d+)")
	posre = re.compile(r"\d+\.\d+")
	rowcol = re.compile(r"\(rows, cols\)\: (?P<rows>\d+) (?P<cols>\d+)")
	skipslices = re.compile(r"Skipping imaging every (?P<slices>\d+)")
	imgexpected = re.compile(r"This will generate (?P<imgs>\d+)")
	acqstopped = re.compile(r"Acquisition cycle stopped after (?P<slices>\d+)")
	dateandtime = re.compile(r"\d\d/\d\d/\d\d\d\d \d\d:\d\d:\d\d [AP]M")
	trimstop = re.compile(r"Stopped trimming after (?P<slices>\d+)")
	trimcomp = re.compile(r"Completed trimming for (?P<slices>\d+)")
	exposure = re.compile(r"Set eposure time to (?P<time>\d+)")
	#initialize variables used to track data about runs and trims
	expectedImgs = None
	skipImg = None
	trimNum = 0
	currentRun = None
	sizeXY = None
	runstart = None
	trimstart = None
	runslices = None
	finalSlice = None
	imagesTaken = None
	#Reads in the file and rips the while thing into a list, ready for parsing
	for row in rawFile:
		if cycletype.search(row):
			t = dateandtime.search(row)
			m = cycletype.search(row)
			m2 = cyclenum.search(row)
			if m['type'] == 'IMAGING':
				#if current run is not None, a new run is starting but the previous run has not yet ended so we end it
				#when this happens we are not able to determine how many images were actually taken
				if currentRun != None:
					runLength[currentRun] = [runslices, runslices, expectedImgs, imagesTaken]
					dates[currentRun] = [runstart, t[0]]
					runslices = None
					finalSlice = None
					expectedImgs = None
					imagesTaken = None
					runstart = None
					currentRun = None
				currentRun = int(m2['cycle'])
				runstart = t[0]
				currentRun = f"{runstart} run {currentRun}"
				history.append(('run', currentRun))
			else:
				trimNum += 1
				trimstart = t[0]
				history.append(('trim', trimNum))
		elif 'XY positions are' in row:
			if currentRun == None:
				print("An error has ocurred, looking for XY positions but run is None")
			xs = {}
			ys = {}
			xys = []
			m = posre.findall(row)
			hpanels = 0
			vpanels = 0
			for i in range(0, len(m), 2):
				x = float(m[i])
				try:
					y = float(m[i+1])
				except IndexError:
					y = float(0)
				if x in xs:
					xs[x] += 1
				else:
					xs[x] = 1
				if y in ys:
					ys[y] += 1
				else:
					ys[y] = 1
				xys.append([x,y])
			XYPositions[currentRun] = xys
			try:
				hpanels = max(ys.values())
			except:
				hpanels = 1
			try:
				vpanels = max(xs.values())
			except:
				vpanels = 1
			panels[currentRun] = [hpanels, vpanels]
		elif 'Z positions are' in  row:
			if currentRun == None:
				print("An error has ocurred, looking for Z positions but run is None")
			ZPositions[currentRun] = [float(x) for x in posre.findall(row)]
		elif 'Pixel size' in row:
			if currentRun == None:
				print("An error has ocurred, looking for XY pixel size but run is None")
			sizeXY = float(posre.findall(row)[0])
			m = rowcol.search(row)
			totalRows = int(int(m['rows']) * (0.8 * panels[currentRun][0] - 0.2 ))
			totalCols = int(int(m['cols']) * (0.8 * panels[currentRun][1] - 0.2 ))
			imageSize[currentRun] = [totalRows, totalCols]
		elif "Skipping imaging every" in row:
			if currentRun == None:
				print("An error has ocurred, looking for skipped slices but run is None")
			if sizeXY == None:
				print("An error has ocurred, looking for skipped slices but xy pixel size is None")
			m = skipslices.search(row)
			skipImg = int(m['slices'])
			sizeZ = 3 * (skipImg + 1)
			voxelSize[currentRun] = (sizeXY, sizeXY, sizeZ)
		elif "This will generate" in row:
			m = imgexpected.search(row)
			expectedImgs = int(m['imgs'])
			if skipImg == None:
				print("An error has ocurred, looking for generated images but skipped slices is None")
			runslices = expectedImgs * (skipImg + 1)
			finalSlice = runslices
		elif "Acquisition cycle stopped after" in row:
			m = acqstopped.search(row)
			t = dateandtime.search(row)
			finalSlice = int(m['slices'])
			imagesTaken = int(runslices/(skipImg + 1))
			if currentRun == None:
				print("An error has ocurred, looking for acquisition stopped but current run is None")
			dates[currentRun] = [runstart, t[0]]
			runLength[currentRun] = [runslices, finalSlice, expectedImgs, imagesTaken]
			runslices = None
			finalSlice = None
			expectedImgs = None
			imagesTaken = None
			runstart = None
			currentRun = None
		elif "Completed acquisition cycle" in row:
			t = dateandtime.search(row)
			if currentRun == None:
				print("An error has ocurred, looking for acquisition completed but current run is None")
			dates[currentRun] = [runstart, t[0]]
			imagesTaken = int(runslices/(skipImg + 1))
			runLength[currentRun] = [runslices, runslices, expectedImgs, imagesTaken]
			runslices = None
			finalSlice = None
			expectedImgs = None
			imagesTaken = None
			runstart = None
			currentRun = None
		elif "Stopped trimming after" in row:
			m = trimstop.search(row)
			t = dateandtime.search(row)
			if trimstart == None:
				print("An error has ocurred, looking for trim length but trim start is None")
			trimLength[trimNum] = [int(m['slices']), trimstart, t[0]]
			trimstart = None
		elif "Completed trimming for" in row:
			m = trimcomp.search(row)
			t = dateandtime.search(row)
			if trimstart == None:
				print("An error has ocurred, looking for trim length but trim start is None")
			trimLength[trimNum] = [int(m['slices']), trimstart, t[0]]
			trimstart = None
		elif "Set eposure time" in row:
			m = exposure.search(row)
			if currentRun == None:
				print("An error has ocurred, looking for acquisition completed but current run is None")
			exposureTime[currentRun] = int(m['time'])
		else:
			continue
	
	masterFile = {'runs':{},'trims':{},'names':{},'panelNumbers':{},'history':[],'runList':[]}
	
	#Compiles all data for the Runs
	for run in panels:
		masterFile['runList'].append(run)
		masterFile['names'][run] = run.split(' ')[-1]
		masterFile['panelNumbers'][run] = panels[run][0] * panels[run][1]
		
		masterFile['runs'][run] = {}
		masterFile['runs'][run]['panels'] = panels[run]
		
		try:
			masterFile['runs'][run]['voxel'] = voxelSize[run]
		except KeyError:
			print(f"KeyError in run {run} for voxelSize")

		try:
			masterFile['runs'][run]['size'] = imageSize[run]
		except KeyError:
			print(f"KeyError in run {run} for ImageSize")


		try:
			masterFile['runs'][run]['exposure'] = exposureTime[run]
		except KeyError:
			print(f"KeyError in run {run} for Exposure Time")

		try:
			masterFile['runs'][run]['length'] = {'total cuts':runLength[run][1],'expected cuts':runLength[run][0],'total images':runLength[run][3],'expected images':runLength[run][2]}
		except KeyError:
			print(f"KeyError in run {run} for Run Length")


		try:
			masterFile['runs'][run]['start'] = dates[run][0]
		except KeyError:
			print(f"KeyError in run {run} for Start Time")

		try:
			masterFile['runs'][run]['end'] = dates[run][1]
		except KeyError:
			print(f"KeyError in run {run} for End Time")
		
		try:
			masterFile['runs'][run]['XY_positions'] = XYPositions[run]
		except KeyError:
			print(f"KeyError in run {run} for XYPositions")
		
		try:
			masterFile['runs'][run]['Z_positions'] = ZPositions[run]
		except KeyError:
			print(f"KeyError in run {run} for ZPositions")
	
	#Compiles all data for the Trimming Cycles
	for run in trimLength:
		masterFile['trims'][run] = {}
		masterFile['trims'][run]['length'] = trimLength[run][0]
		masterFile['trims'][run]['start'] = trimLength[run][1]
		masterFile['trims'][run]['end'] = trimLength[run][2]
	
	masterFile['history'] = history
	return masterFile



class survey:
	def __init__(self,data):
		self.data = data
		self.setup_local_variables_from_data()
		
	def run(self,PATH):
		self.outpath = PATH
		if 'attr' not in self.data:
			self.scrap_all_attr_files_from_zarrs()
		if 'means' not in self.data:
			self.calculate_global_means_for_all_images_and_save_attributes()
			if ms.stopping:
				return
		if 'log' not in self.data:
			self.data['log'] = logFileLoader(self.path)
		if 'shifts' not in self.data:
			self.find_all_alignments_and_set_total_image_size()
		
		self.write_output_images_and_report()
		
	def calculate_global_means_for_all_images_and_save_attributes(self):
		print('Calculating Global Means for all images')
		
		#TRY RECHUNKING DASK ARRAY HERE
		
		self.data['means'] = {}
		self.data['length'] = {}
		self.data['width'] = {}
		self.data['height'] = {}
		self.width = 0
		self.height = 0
		self.length = 0
		for zarrNumber in tqdm(self.allArrays):
			if not self.loadRunFile(zarrNumber): return
			
			self.data['means'][zarrNumber] = np.zeros(self.IMG.length)
			self.data['width'][zarrNumber] = self.IMG.width
			self.data['height'][zarrNumber] = self.IMG.height
			
			if self.width < self.IMG.width:
				self.width = self.IMG.width
			if self.height < self.IMG.height:
				self.height = self.IMG.height
			
			for i in range(self.IMG.length):
				if ms.stopping:return
				img = self.IMG.get_image(i)
				if img is None:
					continue
				self.data['means'][zarrNumber][i] = np.mean(img)
				self.data['length'][zarrNumber] = i
				if self.data['means'][zarrNumber][i] < 1:
					break
			
			self.length += self.data['length'][zarrNumber]
		
		
	
	def compile_report_on_data(self):
		report = {}
		for zarrNumber in self.allArrays:
			MEAN = self.data['means'][zarrNumber][self.data['means'][zarrNumber] > 0]
			report[zarrNumber] = {
				"Run":zarrNumber,
				"Length":self.data['length'][zarrNumber],
				"Width":self.data['width'][zarrNumber],
				"Height":self.data['height'][zarrNumber],
				"Mean Intensity":np.mean(MEAN),
				"Shift":f"{self.shifts[zarrNumber][0]},{self.shifts[zarrNumber][1]}"
			}
		
		report_file = open(self.outpath + "report.csv","w")
		
		for header in report_header:
			report_file.write(header + ',')
		report_file.write('\n')
		
		for zarrNumber in self.allArrays:
			for header in report_header:
				report_file.write(str(report[zarrNumber][header]) + ',')
			report_file.write('\n')
		report_file.close()
	
	def find_all_alignments_and_set_total_image_size(self):
		print("Aligning between all runs...")
		self.survey_images = {}
		self.shifts = {}

		n = len(self.allArrays)
		
		for i in tqdm(range(n)):
			zarrNumber = self.allArrays[i]
			if not self.loadRunFile(zarrNumber): continue

			index = self.data['length'][zarrNumber] - 1
			
			self.survey_images[zarrNumber] = {}
			
			if i == 0:
				img = self.IMG.get_image(0)
				if img is None:
					continue
				self.survey_images[zarrNumber]["first"] = img
				reference = self.survey_images[zarrNumber]["first"]
				self.shifts[zarrNumber] = np.array([0,0]).astype('int64')
				
			else:
				self.survey_images[zarrNumber]["first"] = self.IMG.match_histogram(reference,0)
				pNumber = self.allArrays[i-1]
				self.shifts[zarrNumber] = self.IMG.get_shift_between_images(self.survey_images[pNumber]["last"],self.survey_images[zarrNumber]["first"])
			
			self.survey_images[zarrNumber]["last"] = self.IMG.match_histogram(reference,index)
			self.survey_images[zarrNumber]["resize"] = self.width != self.IMG.width or self.height != self.IMG.height
		self.find_total_mat_size()
		
	def find_total_mat_size(self):
		#Find Minimum Shift Value In array, because Shift Values Can Be Negative
		min_w = 0
		min_h = 0
		max_w = 0
		max_h = 0
		
		self.data['shifts_uncorrected'] = {}
		for zarrNumber in self.shifts:
			self.data['shifts_uncorrected'][zarrNumber] = self.shifts[zarrNumber].tolist()
		
		for zarrNumber in self.allArrays:
			if min_w > self.shifts[zarrNumber][0]:
				min_w = self.shifts[zarrNumber][0]
			if min_h > self.shifts[zarrNumber][1]:
				min_h = self.shifts[zarrNumber][1]
			
			if max_w < self.shifts[zarrNumber][0]:
				max_w = self.shifts[zarrNumber][0]
			if max_h < self.shifts[zarrNumber][1]:
				max_h = self.shifts[zarrNumber][1]
		
		
		#Subtract Minimum Value From All Elements in Shift
		normalize = np.array([min_w,min_h])
		for zarrNumber in self.allArrays:
			self.shifts[zarrNumber] -= normalize
		
		#Computer image size of mat image to be created.
		size_added_w = max_w - min_w
		size_added_h = max_h - min_h
		self.width += size_added_w
		self.height += size_added_h
		
		for zarrNumber in self.shifts:
			self.shifts[zarrNumber] = self.shifts[zarrNumber].tolist()
		
		
		self.data['shifts'] = self.shifts
		self.data['width_survey'] = int(self.width)
		self.data['height_survey'] = int(self.height)
		
	
	def getAttrFromZarr(self,zarrNumber):
		zpath = self.path + 'MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
		data = zarr.open(zpath, mode='r')
		try:
			self.data['attr'][zarrNumber] = data.attrs['multiscales'][0]
		except:
			print("Error in finding zarr attributes file, please review")
			self.data['attr'][zarrNumber] = {}

	def loadRunFile(self,zarrNumber):
		zpath = self.path + 'MUSE_stitched_acq_'  + str(zarrNumber) + '.zarr'
		self.IMG = img.img(zpath,True)
		if self.IMG.failed or ms.stopping:
			print(f"Error zarr file {zpath} is corrupted, please check...")
			return False
		return True

	def setup_local_variables_from_data(self):
		self.allArrays = self.data['runs']
		self.path = self.data['acquire_path']
	

	def scrap_all_attr_files_from_zarrs(self):
		self.data['attr'] = {}
		for zarrNumber in self.allArrays:
			self.getAttrFromZarr(zarrNumber)

	def write_stage_1_movie_pics(self):
		print("Creating flythrough movie for review...")
		directory = self.data["movie_path"] + 'tmp' + os.path.sep
		ms.replace_directory(directory)
	
		self.wIMG = imgc.imageCreator(directory,self.width,self.height,self.length,"png")
		index = 0
		for zarrNumber in tqdm(self.allArrays):
			if not self.loadRunFile(zarrNumber): return
			idx = self.data['length'][zarrNumber]
			for i in range(idx):
				if ms.stopping:return
				img = self.IMG.get_image_with_shift(i,self.width,self.height,self.shifts[zarrNumber],10.0,True,self.data['filename'],True,True,True)
				if img is None:
					continue
				c = ms.format_image_number_to_10000(index)
				fname = f"image_{c}"
				self.wIMG.add_image(img,fname)
				index += 1
		
		self.wIMG.make_flythrough_movie_from_pngs(self.data["movie_path"],"survey_flythrough")
		ms.remove_directory(directory)

	def write_output_images_and_report(self):
		self.write_survey_output_images()
		self.write_stage_1_movie_pics()
		self.compile_report_on_data()
	
	def write_survey_output_images(self):
		print("Writing report and example images...")
		length = 2 * len(self.allArrays)
		
		self.wIMG = imgc.imageCreator(self.outpath,self.width,self.height,length,"png")
		
		if self.wIMG.failed: 
			print("Error, image writing could not find proper output directory. Please check")
			return

		n = len(self.allArrays)
		
		for i in range(n):
			zarrNumber = self.allArrays[i]
			if not self.loadRunFile(zarrNumber): continue

			index = self.data['length'][zarrNumber] - 1
			iImg = self.IMG.get_image_with_shift(0,self.width,self.height,self.shifts[zarrNumber],scalebar = True,reduce_bits = True)
			fImg = self.IMG.get_image_with_shift(index,self.width,self.height,self.shifts[zarrNumber],scalebar = True,reduce_bits = True)
			if iImg is not None:
				self.wIMG.add_image(iImg,f"Run_{zarrNumber}_First")
			if fImg is not None:
				self.wIMG.add_image(fImg,f"Run_{zarrNumber}_Last")


