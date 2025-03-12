# Author: Thrupthi Ann John
# This file runs the code to create all the CIS maps and get quantitative results with the provided CMS map for 100 random images.
# This file is the equivalent of running all 3 demo notebooks. 

import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
import torch
from pathlib import Path
from os.path import join
from skimage.io import imread
from skimage.transform import resize
from scipy.io import loadmat
from CalculateCIS import Occlusion
from PIL import Image
from os import makedirs
from TranslateCMS import TranslateHeatmap, get_image_numpy
from ExplanationMap import SumNormalize, GetExplanationImage

# Define all the folder names
samplefolder = '../data/SampleData'
imagefolder = join(samplefolder, 'SampleImages')
meshfolder = join(samplefolder, 'Sample3D')

resultsfolder = '../results'
cisfolder = join(resultsfolder, 'CIS')

makedirs(cisfolder, exist_ok=True)
modelfolder = '../data/Models'
frontal = imread(join(modelfolder, 'frontal.jpg'))
frontal_mesh = loadmat(join(modelfolder, 'frontal_mesh.mat'))['vertices'][::20,:]
size = 15
device = torch.device('cuda')

gradcamfolder = join(samplefolder, 'GradCAM')
gradcamplusfolder = join(samplefolder, 'GradCAMPlus')
scorecamfolder = join(samplefolder, 'ScoreCAM')
cms = np.load(join(modelfolder, 'Recognition_CMS.npy'))

interfolder = join(resultsfolder, 'IntermediateResults')
transcmsfolder = join(interfolder, 'Translated_CMS')
explcmsfolder = join(interfolder, 'Explanation_CMS')
explgradcamfolder = join(interfolder, 'Explanation_GradCAM')
explgradcamplusfolder = join(interfolder, 'Explanation_GradCAMPlus')
explscorecamfolder = join(interfolder, 'Explanation_ScoreCAM')

makedirs(transcmsfolder, exist_ok=True)
makedirs(explcmsfolder, exist_ok = True)
makedirs(explgradcamfolder, exist_ok = True)
makedirs(explgradcamplusfolder, exist_ok = True)
makedirs(explscorecamfolder, exist_ok = True)

# Define some functions
def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        pil_im.thumbnail((512, 512))
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
   # im_as_var = Variable(im_as_ten, requires_grad=True)
    im_as_ten = im_as_ten.to(device);
    im_as_ten.requires_grad=True;
    return im_as_ten
	
def get_image(filename):
	img = Image.open(filename)
	orig_size = img.size
	img = img.resize((224,224))
	pimg= preprocess_image(img)
	return pimg, orig_size



def Confidence(image, class_index=None):
	with torch.no_grad():
		output = model(image)
	if class_index is None:
		class_index = torch.argmax(output)
	return output[:,class_index].detach(), class_index

def loadVGGModel( filename):
	dat2 = torch.load(filename)
	# copy dictionary
	if str.split(list(dat2.keys())[0],'.')[0] == 'module':
		dat = {}
		for key in dat2.keys():
			k = '.'.join(str.split(key,'.')[1:])
			dat[k] = dat2[key]
	else:
		dat = dat2
		
	n_classes = dat['classifier.6.bias'].shape[0]
	model = models.vgg16(pretrained = False)
	lastlayer = torch.nn.Linear(in_features = model.classifier[-1].in_features, \
							   out_features = n_classes, \
							   bias = True)
	model.classifier[-1] = lastlayer
	model.load_state_dict(dat)
	return model

class Dataset(torch.utils.data.Dataset):
	def __init__(self, datafolder, filelist):
		self.datafolder = datafolder
		self.filelist = filelist
		self.length = len(self.filelist)
		
	def __len__(self):
		return self.length
	
	def __getitem__(self, index):
		return get_image(join(self.datafolder, self.filelist[index]))[0], self.filelist[index]
		
def confidence(dataset):
	dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
	# create variables to store score and class
	score = np.arange(len(dataset)).astype(np.float32)
	index = np.arange(len(dataset))
	filelist = []
	lastind = 0
	for ii, data in enumerate(dataloader):
		img, file = data
		filelist.extend(file)
		print('\r%s'%ii,end='           ')
		outputs = model(img.squeeze().cuda()).detach()
		val, ind = torch.max(outputs,1)
		score[lastind:lastind+len(val)] = val.cpu().numpy()
		index[lastind:lastind+len(ind)] = ind.cpu().numpy()
		lastind = lastind+len(val)
		
	return score, index, filelist

def confidence_for_class(dataset, classes):
	dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=32, shuffle=False)
	# create variables to store score and class
	score = np.arange(len(dataset)).astype(np.float32)
	classes = classes.flatten()
	lastind = 0
	for ii, data in enumerate(dataloader):
		print('\r%s'%ii,end='           ')
		img, filename = data
		outputs = model(img.squeeze().cuda()).detach().cpu().numpy()
		for jj in range(outputs.shape[0]):
			score[lastind:lastind+jj]=outputs[jj][classes[lastind+jj]]
		lastind = lastind+outputs.shape[0]
		
	return score

def AverageDrop(fullscore, explscore):
	return  np.sum(np.maximum(0, fullscore-explscore)/fullscore)*100/len(fullscore)

def IncreaseConfidence(fullscore, explscore):
	return np.sum(fullscore<explscore)/len(fullscore)*100

def Win(gcscore, gcpscore, scscore, trscore):
	# for each, check which one is the hightest
	
	maxscores = np.argmin(np.vstack((gcscore,gcpscore, scscore, trscore)),axis=0)
	gc = np.sum(maxscores==0)
	gcp = np.sum(maxscores==1)
	sc = np.sum(maxscores==2)
	tr = np.sum(maxscores==3)
	length = len(gcscore)
	
	print('GradCAM\t\t:\t', gc/length*100)
	print('GradCAM++\t:\t', gcp/length*100)
	print('ScoreCAM\t:\t', sc/length*100)
	print('CMS\t\t:\t', tr/length*100)

if __name__=="__main__":
    print('Loading the model...')
    # model = loadVGGModel(join(modelfolder, 'VGG16_CelebA_Gender.pth'))
    model = loadVGGModel(join(modelfolder, 'VGG16_CelebA_Recognition.pth')) # here is the recognition model
    model.to(device)
    model.eval()
    model = torch.nn.DataParallel(model)
    
    # get list of files
    p = Path(imagefolder)
    filenames = [i.stem for i in p.glob('**/*.jpg')]

    # get CIS map for each image. This takes some time. 
    print('Creating CIS maps for sample images')
    for ii in range(len(filenames)):
        # print(ii)
        img, sz = get_image(join(imagefolder,filenames[ii]+'.jpg'))
        mesh = loadmat(join(meshfolder, filenames[ii])+'_mesh.mat')['vertices']
        # subsample the mesh to make the calculation faster
        if len(mesh)>2194:
            mesh = mesh[::20,:]
        print(ii, end=', ',flush=True )
        heatmap = Occlusion(Confidence, img.to(device, dtype = torch.float), mesh, sz, frontal, frontal_mesh, size = size,class_index=None, device = device);
        outfilename = join(cisfolder, filenames[ii])
        np.save(outfilename, heatmap)
     
    # Generate the CMS file and save it to results folder
    print('\nGenerate the CMS file...')
    p = Path(cisfolder)
    filenames = [i.stem for i in p.glob('**/*.npy')]
    heatmap = []
    for ii in range(len(filenames)):
        hmap = np.load(join(cisfolder, filenames[ii]+'.npy'))
        if len(heatmap)==0:
            heatmap = hmap
        else:
            heatmap = heatmap+hmap
    np.save(join(resultsfolder, 'Generated_CMS'),heatmap)
    print('CMS file generated at ', join(resultsfolder, 'Generated_CMS.npy'))
    
    # Translate CMS maps
    print('Translate the CMS maps...')
    for ii in filenames:
        pimg, sz = get_image_numpy(join(imagefolder, ii+'.jpg'))
        mesh = loadmat(join(meshfolder, ii+'_mesh.mat'))['vertices']
        if len(mesh)>2194:
            mesh = mesh[::20,:]
        newheatmap = TranslateHeatmap(cms, frontal_mesh, mesh, pimg, sz)
        np.save(join(transcmsfolder, ii+'.npy'), newheatmap)
        
    # Calculate explanation maps
    print('Calculate the explanation maps... ')
    for ii in filenames:
        img = resize(imread(join(imagefolder, ii+'.jpg')),(224,224))
        cis = resize(np.load(join(transcmsfolder, ii+'.npy')),(224,224))
        gradcam  = resize(np.load(join(gradcamfolder, ii+'.npy')),(224,224))
        gradcamplus = resize(np.load(join(gradcamplusfolder, ii + '.npy')),(224,224))
        scorecam = resize(np.load(join(scorecamfolder, ii+'.npy')),(224,224))
        gradcam = SumNormalize(gradcam, cis)
        gradcamplus = SumNormalize(gradcamplus, cis)
        scorecam = SumNormalize(scorecam, cis)
        cis = SumNormalize(cis, cis)
        GetExplanationImage(img, 1-cis, join(explcmsfolder, ii+'.jpg'))
        GetExplanationImage(img, 1-gradcam, join(explgradcamfolder, ii+'.jpg'))
        GetExplanationImage(img, 1-gradcamplus, join(explgradcamplusfolder, ii+'.jpg'))
        GetExplanationImage(img, 1-scorecam, join(explscorecamfolder, ii+'.jpg'))
        
    print('Calculating scores... (final step)s')
    # score on unaltered images
    filenames = [i.stem + '.jpg' for i in p.glob('**/*')]
    images = Dataset(imagefolder, filenames)
    imagescore, index, filelist = confidence(images)

    # create other datasets with the same file order
    gradcam = Dataset(explgradcamfolder, filelist)
    gradcamplus = Dataset(explgradcamplusfolder, filelist)
    scorecam = Dataset(explscorecamfolder, filelist)
    cms = Dataset(explcmsfolder, filelist)

    # get scores for the original classes
    gradcamscore = confidence_for_class(gradcam, index)
    gradcamplusscore = confidence_for_class(gradcamplus, index)
    scorecamscore = confidence_for_class(scorecam, index)
    cmsscore = confidence_for_class(cms, index)

    print('Average Drop: (Higher is better)')
    print('GradCAM\t\t:\t', AverageDrop(imagescore, gradcamscore))
    print('GradCAM++\t:\t', AverageDrop(imagescore, gradcamplusscore))
    print('ScoreCAM\t:\t', AverageDrop(imagescore, scorecamscore))
    print('CMS\t\t:\t', AverageDrop(imagescore, cmsscore))
    print('\n% Increase in confidence: (Lower is better)')
    print('GradCAM\t\t:\t', IncreaseConfidence(imagescore, gradcamscore))
    print('GradCAM++\t:\t', IncreaseConfidence(imagescore, gradcamplusscore))
    print('ScoreCAM\t:\t', IncreaseConfidence(imagescore, scorecamscore))
    print('CMS\t\t:\t', IncreaseConfidence(imagescore, cmsscore))
    print('\nWin %: (Higher is better)')
    Win(gradcamscore, gradcamplusscore, scorecamscore, cmsscore)
    
    