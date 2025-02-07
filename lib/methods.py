import os, shutil

#Variables for GUI interaction
stopping = False #Stopping lets the GUI stop all long processes in the program.


#Image viewer variables
shift_crop_size = 1000 #Crop size when calculating shifts
max_shift_x = 2500 #maximum allowed shift amount in height
max_shift_y = 3500 #maximum allowed shift amount in width

#Image Creator Variables
spacing = 500
scalebar_color = (255,255,255)


def copy(variable):
	return variable

def format_image_number_to_10000(c):
	if c < 10:
		text = '000' + str(c)
	elif c < 100:
		text = '00' + str(c)
	elif c < 1000:
		text = '0' + str(c)
	else:
		text = str(c)
	return text

def make_directory(directory):
	if not os.path.isdir(directory):
		os.makedirs(directory)

def replace_directory(directory):
	if os.path.isdir(directory):
		shutil.rmtree(directory)
	os.makedirs(directory)

def remove_directory(directory):
	if os.path.isdir(directory):
		shutil.rmtree(directory)


