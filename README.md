# MUSE-preprocessing
This is a repository for 3D-MUSE preprocessing software

Data Surveyor Needs to be run first to generate metadata files. 
Command: python data_surveyor.py <File Name> <Path to Data> <Options>

--o <path> 		Changes output directory 
--su 		Perform Data Survey 
--r 		Finds all folders in the base directory 
--f 		Creates a video flythrough (Linux Only) 
--i 		Saves the invidual curves for means, variance, and difference 




This is a help for Seckler Post Processing Software.
It expects to accept the input from MUSE REVA Preprocessing Software.
Command: python post_processing.py <File Name> <Path to Data> <Run Array> <Options>

--bk <break point> 		Skips to image listed, analyzes that image, and that stops program. Default 0th image

--bt 		Preforms enhanced contrast enhancement using TopHat and BlackHat Imaging Modalities 

--cp <height min> <height max> <width min> <width max> 		Crops the image to the specified height and width. Default: Will not crop 

--ct <contrast factor> 		Contrasts the data according to new_px = factor * (old_px - mean) + 2055. Default: Factor = 1 

--d <scaling factor> 		Downscale data by whatever factor the user inputs. Default: 4 

--h 		Generates and prints help message 

--m <mean> <std> 		Override mean to save time, you must input the mean and the standard deviation as integers of floating point values 

--n <run start> <run end> <background min height> <background max height> 		Normalizes the background of a run when the light was misaligned 

--o <path> 		Changes output directory 

--p <processors> 		Sets the maximum number of processing cores that program will used. Default is 4 

--sb 		Adds scalebar to images outputed 

--sk 		Skips aligning between runs 

--v 		Makes a set of downscaled pngs to be compiled into a flythrough in /flythrough/ 

--z 		Write output to zarr file rather than pngs 
