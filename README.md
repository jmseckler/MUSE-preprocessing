# MUSE-preprocessing
This is a repository for 3D-MUSE preprocessing software

This is a help for Seckler Data Surveyor Software.
This is the first step in data processing and compiles all of the metadata for the MUSE software. It accepts the direct output from MUSE Acquire.
Command: python data_surveyor.py <File Name> <Path to Data> <Options>

--8b 		Downsamples zarr to 8-bit data 
--br <intensity> 		Changes the global brightness of the images, Default: 1000 
--bk <Index> 		Only take out one image as a PNG and ignores all others 
--bt <kernel> 		Performs Black Hat and Top Hate Contrasting, kernel size can be set 
--ma <type> <kernel> 		Choose a modality to add to the image, often paired with -ms to subtract another modality. Of the form Modality type: Opening, Closing, Gradient, TopHat,BlackHat, and a kernal size Default: None, 50 pixels 
--ms <type> <kernel> 		Choose a modality to subtract from the image, often paired with -ma to add another modality. Of the form Modality type: Opening, Closing, Gradient, TopHat,BlackHat, and a kernal size Default: None, 50 pixels 
--d <kernel> 		Dialates image features. Default: 2 pixels 
--e <kernel> 		Erodes image features, good for elucidating axons. Default: 2 pixels 
--png 		Saves final output as PNG file 
--c <arrays> 		Surveys data, collects all metadata, and outputs intial files 
--f 		Creates a video flythrough (Linux Only) 
--fg <spacing> 		Creates a video flythrough with gridlines overlayed 
--fi <crop points> 		Crops and histogram matches all data after cropping, this accepts a list [Height Min, Highe Max, Width Min, Width Max] 
--i 		Saves the invidual curves for means, variance, and difference 
--p 		Process Image 
--o <path> 		Changes output directory 
--of <min> 		Sets boundaries for focus exclusion 
--os <min> <max> 		Sets boundary for similarity exclusion 
--r 		Finds all folders in the base directory 
--s 		Surveys data, collects all metadata, and outputs intial files 
--sk <alignments> 		Skips aligning between runs, assumes 0 shift for all shifts not entered. Enter shift in form of xshift,yshift for each array 
--su 		Rewrites Data Surveyor Metadata Files 
--tr <amount> 		Rewrites Data Surveyor Metadata Files 
--w 		Windows data and requires variables of the form <Min Intensity> <Max Intensity> 
