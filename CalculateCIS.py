#Author: Thrupthi Ann John https://github.com/ThrupthiAnn

import torch
from scipy.io import loadmat
import math


class OData(torch.utils.data.Dataset):
	def __init__(self, image, vertices, orig_img_size, size=15):
		self.image = image
		self.vertices = vertices
		self.size = size
		self.length = vertices.shape[0]
		self.img_row = image.shape[2]
		self.img_col = image.shape[3]
		# normalize points
		self.vertices[:,0] = vertices[:,0]*self.img_row/orig_img_size[0]
		self.vertices[:,1] = vertices[:,1]*self.img_col/orig_img_size[1]

	def show(self, index):
		item = self.__getitem__(index)[0]
		img = recreate_image(torch.unsqueeze(item.cpu(),0))
		plt.imshow(img)
		plt.show()

	def __len__(self):
		return self.length

	def __getitem__(self, index):
	#         set_trace()
		x1 = max(0,math.floor(self.vertices[index, 0]-self.size/2))
		x2 = min(x1+self.size, self.img_row)
		y1 = max(0,math.floor(self.vertices[index, 1]-self.size/2))
		y2 = min(y1+self.size, self.img_col)
		pimg = self.image.clone()
		pimg[:,:,y1:y2, x1:x2] = 0
		return torch.squeeze(pimg), index
        

def Occlusion(confidence, image, vertices, orig_img_size,  frontal_img, frontal_vertices, class_index=None, size=15,  device = torch.device('cuda'), batchsize=16):
    
	# initialize
	if len(image.shape)==3:
		image= image.unsqueeze(0)
	original, class_index = confidence(image)
	img_row = frontal_img.shape[0]
	img_col = frontal_img.shape[1]
	frontsqsz = size*img_row/image.size()[3]
	heatmap = torch.zeros((img_row, img_col))
	heatmap = heatmap.to(device)
	count = torch.zeros((img_row, img_col))
	count = count.to(device)
	dataset = OData(image, vertices, orig_img_size, size, )
	loader = torch.utils.data.DataLoader(dataset,  batch_size = torch.cuda.device_count()*batchsize, shuffle=True)

	for ii, data in enumerate(loader):
		images = data[0]
		indices = data[1]
		output, _ = confidence(images, class_index)
		diff = original - output
		for jj in range(len(output)):
			x1 = max(0,math.floor(frontal_vertices[indices[jj], 0]-frontsqsz/2))
			x2 = min(math.floor(x1+frontsqsz), img_row)
			y1 = max(0,math.floor(frontal_vertices[indices[jj], 1]-frontsqsz/2))
			y2 = min(math.floor(y1+frontsqsz), img_col)
			heatmap[y1:y2, x1:x2] += diff[jj]
			count[y1:y2, x1:x2]+=1
			heatmap = heatmap.detach()
			count = count.detach()

	# normalize for count
	heatmap[count==0] = 0
	count[count==0] = 1
	heatmap = heatmap/count

	return heatmap.cpu()


