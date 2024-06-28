# MUSE-preprocessing
This is a repository for 3D-MUSE preprocessing software

This is a help for Seckler Post Processing Software.
It expects to accept the input from MUSE REVA Preprocessing Software.
Command: python post_processing.py <File Name> <Path to Data> <Fire Run> <Last Run> <Options>

-bk <image number>	Skips to image listed, analyzes that image, and that stops program. Default 0th image
-bt			Preforms enhanced contrast enhancement using TopHat and BlackHat Imaging Modalities
-ct <factor>		Contrasts the data according to new_px = factor * (old_px - mean) + 2055. Default: Factor = 3 and Mean = Image Mean
-cp <height_min> <height_max> <width_min> <width_max>	Crops the image to the specified height and width. Default: Will not crop
-d <scale>		Downscale data by whatever factor the user inputs. Default: 5
-m <mean> <std>		Override mean to save time
-n <run start> <run end> <background min heigh> <background max height>	Normalizes the background of a run when the light was misaligned
-z			Write output to zarr file rather than pngs
-h:			Prints Help Message
-sb			Adds scalebar to images outputed
