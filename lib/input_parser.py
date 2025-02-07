#Import Libraries
import os, ast
import numpy as np


#Define All Used Variables Here
cmdInputs = {
	'-c':{"name":"Finish Data","base":"compile_data","types":['list'],"names":['data_quality'],"variable":[[]],"active":False,"tooltips":"Surveys data, collects all metadata, and outputs intial files"},
	'-nr':{"name":"Override Recursive","base":"recursive_override","types":[],"names":[],"variable":[],"active":False,"tooltips":f"Overrides automatic recursive file calculations. This will process the first file it finds and no others"},
	'-o':{"name":"Override Output","base":"output","types":['str'],"names":['outpath'],"variable":[f".{os.path.sep}output{os.path.sep}"],"active":False,"tooltips":"Changes output directory"},
	'-png':{"name":"Finish Data","base":"png","types":['int'],"names":['png_index'],"variable":[-1],"active":False,"tooltips":"Saves final stage as png stack, if index is given, saves only a single file"},
	'-s':{"name":"Survey Data","base":"survey","types":[],"names":[],"variable":[],"active":False,"tooltips":"Surveys data, collects all metadata, and outputs intial files"}
	}


class inputs():
	def __init__(self,cmd,path = None):
		self.cmdInputs = cmdInputs # Reads in baseline command tags which can be used
		self.inputParser(cmd) # Parses command line inputs setting default values where they need to be set
		
		if path is not None:
			self.compile_survey(path)
	
	def generateHelpString(self,tag):
		entry = self.cmdInputs[tag]
		helpString = ''
		helpString += '-' + tag + ' '
		for e in entry['names']:
			helpString += '<' + e + '> '
		helpString += '		'
		helpString += entry['tooltips'] + ' '
		return helpString

	def inputParser(self,cmd):
		n = len(cmd)
	
		if n < 2 and '-h' not in cmd:
			print("No filename and/or path given...")
			self.printHelp()
			quit()
	
		if '-h' in cmd:
			self.printHelp()
		
		if n < 2 and '-h' not in cmd:
			print("No filename and/or path given...")
			self.printHelp()
			quit()
	
		if '-h' in cmd:
			self.printHelp()
	
		for i in range(n):
			tag = cmd[i]
			if tag[0] == "-" and tag in self.cmdInputs:
				self.cmdInputs[tag]['active'] = True
				
				m = len(self.cmdInputs[tag]['names'])
				for j in range(m):
					try:
						inputValue = cmd[i + j + 1]
						self.cmdInputs[tag]['variable'][j] = self.parse_inputs(inputValue,self.cmdInputs[tag]['types'][j])
					except:
						print(f"Input {tag} has failed to read in input values, using defaults...")
			elif tag == '-jms':
				self.path = '/media/' + getpass.getuser() + '/' + cmd[1] + '/data/'
	
		self.path = cmd[1]
		
		if not self.path.endswith(os.path.sep):
			self.path += os.path.sep
		if not self.path.startswith(os.path.sep) and not self.path.startswith('.'):
			self.path = os.path.sep + self.path
	
	def parse_inputs(self,value,tag):
		match tag:
			case 'float':
				inputValue = float(inputValue)
			case 'str':
				if value[0] != '-':
					inputValue = value
			case 'list':
				inputValue = ast.literal_eval(value)
			case 'slist':
				inputValue = value.split(',')
			case _:
				inputValue = None
		return inputValue
	
	def compile_survey(self,inpath):
		self.compile = {
			"crop":{"height":[0,-1],"width":[0,-1],"total":[0,-1,0,-1]},
			"runs":{},
			"success":True
			}
		surveyPath = inpath + "survey_form.csv"
		if os.path.isfile(surveyPath):
			rawfile = open(surveyPath,'r')
			for row in rawfile:
				if self.compile["success"]:
					header = row.split(',')[0]
					if header.startswith(("Crop Height","Crop Width","Run_")):
						self.record_survey_data(row)
		else:
			print(f"Survey file, {surveyPath} , not found, assuming file does not exist...")
			self.compile["success"] = False
		
		self.compile_the_compiled_survey()
		
	def compile_the_compiled_survey(self):
		self.compile['shifts'] = {}
		
		for zarrNumber in self.compile['runs']:
			self.compile['shifts'][zarrNumber] = self.compile['runs'][zarrNumber]['shift']
		
		self.compile['crop']['total'] = [self.compile['crop']['height'][0],self.compile['crop']['height'][1],self.compile['crop']['width'][0],self.compile['crop']['width'][1]]
		

	def printHelp(self):
		print("This is a help for Seckler Data Surveyor Software for MUSE Acquire Data.")
		print("This is the first step in data processing and compiles all of the metadata for the MUSE software. It accepts the direct output from MUSE Acquire or from MUSE Processor.")
		print("Command: python data_surveyor.py <Path to Data> <Options>")
		print("")
		for entry in self.cmdInputs:
			print(self.generateHelpString(entry))
		quit()
	
	def record_survey_data(self,row):
		line = row.split(',')
		match line[0]:
			case "Description":
				self.record_survey_data_basic(line,"description")
			case "Stain":
				self.record_survey_data_basic(line,"stains")
			case "Counterstain":
				self.record_survey_data_basic(line,"counterstains")
			case "Crop Height":
				self.record_survey_data_crop(line,"height")
			case "Crop Width":
				self.record_survey_data_crop(line,"width")
			case _:
				self.record_survey_data_crop_runs(line)

	def record_survey_data_crop(self,data,crop):
		try:
			self.compile['crop'][crop] = [int(data[1]),int(data[2])]
		except ValueError:
			print(f"Crop {crop.capitalize()} contains invalid values, please correct...")
			self.compile["success"] = False

	def record_survey_data_crop_runs(self, data):
		zarrNumber = data[0].split("_")[-1]
		try:
			self.compile['runs'][zarrNumber] = {"type":int(data[1]),"shift":np.array([int(data[3]),int(data[2])])}
		except ValueError:
			print(f"Run #{zarrNumber} contains invalid values, please correct...")
			self.compile["success"] = False

	def record_survey_data_basic(self, data,tag):
		zarrNumber = data[0].split("_")[-1]
		try:
			self.compile[tag][zarrNumber] = [data[1],data[2],data[3]]
		except ValueError:
			print(f"Run #{zarrNumber} contains invalid values, please correct...")
			self.compile["success"] = False



