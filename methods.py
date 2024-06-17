import glob, shutil, os, zarr
import tifffile as tiffio

from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscale
import dask.array as da

downsample = 1
elipse_size = 30
contrast_factor = 1
nerve_factor = 0
mean = 2047

crop = [0,-1,0,-1]


zarr_attr_path = "./.zattrs"

def find_unprocessed_data_folders(path,outpath):
	if not os.path.isdir(outpath):
		os.makedirs(outpath)
	
	data_list = glob.glob(path + "*")
	finished_paths = glob.glob(outpath + "*")
	flist = []
	finished_list = []
	
	for pname in finished_paths:
		fname = pname.split("/")[-1]
		finished_list.append(fname)
	
	for pname in data_list:
		fname = pname.split("/")[-1]
		if fname not in finished_list:
			flist.append(pname)
	return flist

def zarr_image_lister(path,etx="zarr"):
	path = path + "/M*"
	raw = glob.glob(path)
	
	flist = []
	for fname in raw:
		if fname.endswith((etx)):
			flist.append(fname)
	flist = sorted(flist)
	return flist

def get_image_from_zarr(path):
	zfile = zarr.open(path)
	try:
		return zfile["/muse/stitched/"], zfile.attrs
	except KeyError:
		print(f"Filename, {path} does not exist or is corrupted returning tiff stack instead...")
		return get_image_from_tiff(path), None

def path_number_parser(path):
	run_num = path.split(".")[0]
	run_num = run_num.split("_")[-1]
	try:
		run_num = int(run_num)
	except ValueError:
		run_num = 0
	
	return run_num

def get_image_from_tiff(zarr_path):
	run_num = path_number_parser(zarr_path)
	path_parts = zarr_path.split("/")
	path_parts.pop()
	
	path = ""
	for p in path_parts:
		path += p + "/"
	path += "MUSE_acq_" + str(run_num)
	
	if not os.path.isdir(path):
		print(f"Filename, {path} does not exist or is corrupted returning nothing...")
		return None
	
	tiff_list = zarr_image_lister(path,"tif")
	tiff_list = sorted(tiff_list,key=path_number_parser)
	
	n = find_tiff_image_length(tiff_list)
	
	try:
		tmp = tiffio.imread(tiff_list[0])
	except:
		return None
	
	x = tmp.shape[-2]
	y = tmp.shape[-1]
	
	if len(tiff_list) == 0:
		print(f"Filename, {path} does not exist or is corrupted returning nothing...")
		return None
	
	tiff_zarr = create_basic_zarr_file(path,"data")
	data = tiff_zarr.zeros('data', shape=(n, x, y), chunks=(4, x, y), dtype="i2" )
	
	counter = 0
	for tname in tiff_list:
		length = get_tiff_stack_length(tname)
		for i in range(length):
			data[counter] = tiffio.imread(tname, key = int(i))
			counter += 1
	
	return data
	
def convert_zarr_to_napari_readable(path,zname):
	zfile = da.from_zarr(path + zname + '_tmp.zarr',component="/data")
	
	p = [zfile["/data/data/"], zfile["/data/down5/"], zfile["/data/down10/"]]
	multiscales, ct = build_coordinate_transforms_metadata_transformation_from_pixel()
	
	store_real = parse_url(path + zname + '.zarr', mode="w").store
	root_real = zarr.group(store_real)
	
	write_multiscale(pyramid=p, group=root_real,  axes=["z", "y", "x"], coordinate_transformations = ct)
	
	#manually fix multiscales metadata to inlude units for axes
	root_real.attrs["multiscales"] = multiscales
	
	p_seg = [zfile["/data/segment/"],zfile["/data/segment5x"],zfile["/data/segment10x"]]
	
	store_seg = parse_url(path + zname + '_seg.zarr', mode="w").store
	root_seg = zarr.group(store_seg)
	write_multiscale(pyramid=p_seg, group=root_seg,  axes=["z", "y", "x"], coordinate_transformations = ct)
	root_seg.attrs["multiscales"] = multiscales
	
	remove_directory(path + zname + '_tmp.zarr')
	
	

def find_tiff_image_length(tiff_list):
	count = 0
	for tname in tiff_list:
		count += get_tiff_stack_length(tname)
	return count

def get_tiff_stack_length(tname):
	try:
		with tiffio.TiffFile(tname) as tif:
			return len(tif.pages)
	except:
		return 0


def create_basic_zarr_file(path,fname):
	zarr_path = path + '/' + fname + '.zarr'
	if os.path.isdir(zarr_path):
		shutil.rmtree(zarr_path)
	store = zarr.DirectoryStore(zarr_path, dimension_separator='/')
	root = zarr.group(store=store, overwrite=True)
	data = root.create_group('data')
	return data
	


def replace_directory(directory):
	remove_directory(directory)
	os.makedirs(directory)

def remove_directory(directory):
	if os.path.isdir(directory):
		shutil.rmtree(directory)

def copy_directory(src,dst):
	shutil.copytree(src,dst)

def copy_zarr_attr(path,zname):
	dst = path + '/' + zname + '.zarr/.zattrs'
	shutil.copyfile(zarr_attr_path, dst)

def build_coordinate_transforms_metadata_transformation_from_pixel():
	ct1 = [{"type": "scale", "scale": [9, .9, .9]}]
	ct5 = [{"type": "scale", "scale": [9, 4.5, 4.5]}]
	ct10 = [{"type": "scale", "scale": [9, 9, 9]}]
	
	ct = [ct1, ct5, ct10]
	
	#build multiscales metadata
	axes = [
		{"name": "z", "type": "space", "unit": "micrometer"},
		{"name": "y", "type": "space", "unit": "micrometer"},
		{"name": "x", "type": "space", "unit": "micrometer"}]
	
	datasets = [
		{"coordinateTransformations": ct1, "path": "0"},
		{"coordinateTransformations": ct5, "path": "1"},
		{"coordinateTransformations": ct10, "path": "2"}]
	
	multiscales = [{
		"name": "/data",
		"version": "0.4",
		"axes": axes,
		"datasets": datasets,
	}]
	
	return multiscales, ct

