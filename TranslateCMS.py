#Author: Thrupthi Ann John https://github.com/ThrupthiAnn

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import glob
import scipy.io
import skimage.io
import argparse
import pdb

def get_image_numpy(filename):
    img = Image.open(filename)
    orig_size = img.size
    img = img.resize((224,224))
    img = np.array(img)
    return img, orig_size

def TranslateHeatmap(heatmap, frontal_mesh, mesh, img, orig_sz):
        img_row = img.shape[0]
        img_col = img.shape[1]
        new_heatmap = np.zeros((img_row, img_col))
        count = np.zeros((img_row, img_col))
        frontsqsz = 15*img_row/img.shape[0]
        xfactor = img_row/orig_sz[0]
        yfactor = img_col/orig_sz[1]
        mesh = mesh.copy()
        mesh[:,0] = mesh[:,0]*xfactor
        mesh[:,1] = mesh[:,1]*yfactor
        for ii in range(frontal_mesh.shape[0]):
            x1 = max(0,math.floor(mesh[ii, 0]-frontsqsz/2))
            x2 = min(math.floor(x1+frontsqsz), img_row)
            y1 = max(0,math.floor(mesh[ii, 1]-frontsqsz/2))
            y2 = min(math.floor(y1+frontsqsz), img_col)
            new_heatmap[y1:y2, x1:x2] += heatmap[math.floor(frontal_mesh[ii][1]), math.floor(frontal_mesh[ii][0])];
            count[y1:y2, x1:x2]+=1
            #new_heatmap[math.floor(mesh[ii][1]), math.floor(mesh[ii][0])] = heatmap[math.floor(frontal_mesh[ii][1]), math.floor(frontal_mesh[ii][0])];
        new_heatmap[count==0] = 0
        count[count==0] = 1
        new_heatmap = new_heatmap/count
        return new_heatmap
	
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file_list', type=str, help="Specify path of txt file containing list of images for Grad-CAM++ visualization")
	parser.add_argument('-i', '--image_folder', default='.', type=str, help="Specify folder which has input images, default = '.'")
	parser.add_argument('-m', '--mesh_folder', default='.', type=str, help='Specify folder which has corresponding input meshes')
	parser.add_argument('-o', '--output_folder',default='.', type=str, help="Specify output folder for visualization, default current folder ")
	parser.add_argument('-c', '--cms', type=str, help="Specify location of the CMS map")
	args = parser.parse_args()
	
	# get the frontal mesh and cms
	frontal_mesh = scipy.io.loadmat('Models/frontface/frontal_mesh.mat')['vertices'][::20,:]
	cms = np.load(args.cms)
	
	file = open(args.file_list, 'r')
	inputlist = file.read().splitlines()
	file.close()
	for ii in inputlist:
		filename = os.path.join(args.image_folder, ii) 
		pimg, sz = get_image_numpy(filename)
		root = os.path.splitext(ii)[0]

		filename = os.path.join(args.mesh_folder, root+'_mesh.mat')
		mesh = scipy.io.loadmat(filename)['vertices']

		newheatmap = TranslateHeatmap(cms, frontal_mesh, mesh, pimg, sz)
		outfilename = os.path.join(args.output_folder, root)
		np.save(outfilename, newheatmap)