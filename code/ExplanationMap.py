# Author: Thrupthi Ann John https://github.com/ThrupthiAnn

import numpy as np
import skimage.io
import skimage.transform
import skimage.filters
import argparse
import os
from PIL import Image
import pdb

def GetExplanationImage(img, cam, filename):
	img0 = np.expand_dims(img[:,:,0]*cam,2)
	img1 = np.expand_dims(img[:,:,1]*cam,2)
	img2 = np.expand_dims(img[:,:,2]*cam,2)
	newimg = np.concatenate((img0,img1,img2),2)
	newimg = Image.fromarray((newimg*255).astype(np.uint8))
	newimg.save(filename)
	return newimg

def ImageMultiply(img, cam):
	img0 = np.expand_dims(img[:,:,0]*cam,2)
	img1 = np.expand_dims(img[:,:,1]*cam,2)
	img2 = np.expand_dims(img[:,:,2]*cam,2)
	newimg = np.concatenate((img0,img1,img2),2)
	return newimg

def GetBlurImage(img,cam, filename, sigma = 3.0):
	blurred = skimage.filters.gaussian(img, sigma=(sigma,sigma), truncate=3.5, multichannel=True)
	cam = (cam-np.min(cam))/(np.max(cam)-np.min(cam))
	newimg = ImageMultiply(img, cam) + ImageMultiply(blurred, 1-cam)
	newimg = Image.fromarray((newimg*255).astype(np.uint8))
	newimg.save(filename)
	return newimg

def SpecialNormalize(heatmap, standard):
	numpositive = np.sum(standard>0)
	x = np.sort(heatmap.flatten())[-numpositive]
	heatmap = (heatmap-x)/(heatmap.max()-x)
	heatmap = np.maximum(heatmap,0)
	return heatmap

def SumNormalize(heatmap, standard):
	# clip standard and heatmap
	standard = np.maximum(standard,0)
	heatmap = np.maximum(heatmap,0)
	
	standard = (standard-np.min(standard))/(np.max(standard)-np.min(standard))
	heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap)-np.min(heatmap))
	
	stdsum = np.sum(standard)
	hmsum = np.sum(heatmap)
	
	hm = heatmap * stdsum/hmsum
	return hm

def Normalize(heatmap):
	hmap = (heatmap - np.min(heatmap))/(np.max(heatmap)-np.min(heatmap))
	return hmap

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file_list', type=str, help="Specify path of txt file containing list of images for Grad-CAM++ visualization")
	parser.add_argument('-o', '--output_folder',default='.', type=str, help="Specify output folder for explanation maps, default . ")
	parser.add_argument('-i', '--image_folder', default = '.', type=str, help = 'Specify the folder containing images')
	parser.add_argument('-s', '--standard_hmap_folder', type=str, help='Specify the folder containing standard heatmaps, only needed if using special normalization')
	parser.add_argument('-p', '--heatmap_folder', type=str, help='Specify the folder containing heatmaps')
	parser.add_argument('-n', '--special_normalization', default=False, action='store_true',  help='Specify if you want to use special normalization, default False')
	args = parser.parse_args()
	
	file = open(args.file_list, 'r')
	inputlist = file.read().splitlines()
	file.close()
	jj=0
	for ii in inputlist:
		jj = jj+1
		if jj%100==0:
			print('%d of %d'%( jj, len(inputlist)))
		root = os.path.splitext(ii)[0]
		img = skimage.io.imread(os.path.join(args.image_folder, root+'.jpg'))
		img = skimage.transform.resize(img, (224,224))
		heatmap = np.load(os.path.join(args.heatmap_folder, root+'.npy'))
		heatmap = skimage.transform.resize(heatmap, (224,224))
		if args.special_normalization:
			stdmap = np.load(os.path.join(args.standard_hmap_folder, root+'.npy'))
			stdmap = skimage.transform.resize(stdmap, (224,224))
			heatmap = SumNormalize(heatmap, stdmap)
		else:
			heatmap = Normalize(heatmap)
			
		expimg = GetExplanationImage(img, 1-heatmap,os.path.join(args.output_folder, root+'.jpg') )
		pass;
