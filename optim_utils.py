
from PIL import Image
import os
from torchvision import transforms    
import torch
import numpy as np
import torch.nn as nn

# load the input images and ground truth light position
def load_input(image_dir, light_dir, scene):
	# load images
	for i in range(9):
		image_i = os.path.join(image_dir,'{}_{}.png'.format(scene,i)) 
		image_i = Image.open(image_i).convert('RGB')
		image_i = transforms.ToTensor()(image_i).permute(1,2,0).cuda() 
		# save_image(util.tensor2im(image_i), join(img_dir,'input_{}.jpg'.format(i)))

		TrainData = torch.cat((TrainData,image_i.unsqueeze(0)),dim=0) if i!=0 else image_i.unsqueeze(0)

	# load lights
	LightPos = torch.from_numpy(load_light_txt(os.path.join(light_dir,'MGReal9/{}.txt'.format(scene)))).float().cuda()

	return TrainData.permute(0,3,1,2), LightPos.unsqueeze(-1).unsqueeze(-1)


def load_light_txt(name):
	with open(name,'r') as f:
		lines = f.readlines()
		wlvs = []
		for line in lines[0:]:
			line = line[:-1]
			camera_pos = [float(i) for i in line.split(',')]
			wlvs.append(camera_pos)
		wlvs=np.array(wlvs)

		print(wlvs)
		return wlvs


class VGGLoss(nn.Module):
	def __init__(self, gpu_ids, gt):
		super(VGGLoss, self).__init__()        
		self.vgg = Vgg19().cuda()
		self.criterion = nn.L1Loss()
		self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        
		self.gt_vgg = self._gt_vgg(gt)

	def forward(self, x):              
		x_vgg = self.vgg(self._preprocess(x))
		loss = 0
		for i in range(len(x_vgg)):
			loss += self.weights[i] * self.criterion(x_vgg[i], self.gt_vgg[i].detach())   
		return loss

	def _gt_vgg(self,img):
		img_vgg = self.vgg(self._preprocess(img))
		# loss = 0
		# for i in range(len(x_vgg)):
		# 	loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())   
		return img_vgg

	def _preprocess(self,x):  
		mean = torch.tensor([0.485, 0.456, 0.406]).cuda().unsqueeze(-1).unsqueeze(-1)
		std = torch.tensor([0.229, 0.224, 0.225]).cuda().unsqueeze(-1).unsqueeze(-1)
		return (x-mean)/std

from torchvision import models
class Vgg19(torch.nn.Module):
	def __init__(self, requires_grad=False):
		super(Vgg19, self).__init__()
		vgg_pretrained_features = models.vgg19(pretrained=True).features
		self.slice1 = torch.nn.Sequential()
		self.slice2 = torch.nn.Sequential()
		self.slice3 = torch.nn.Sequential()
		self.slice4 = torch.nn.Sequential()
		self.slice5 = torch.nn.Sequential()
		for x in range(2):
			self.slice1.add_module(str(x), vgg_pretrained_features[x])
		for x in range(2, 7):
			self.slice2.add_module(str(x), vgg_pretrained_features[x])
		for x in range(7, 12):
			self.slice3.add_module(str(x), vgg_pretrained_features[x])
		for x in range(12, 21):
			self.slice4.add_module(str(x), vgg_pretrained_features[x])
		for x in range(21, 30):
			self.slice5.add_module(str(x), vgg_pretrained_features[x])
		if not requires_grad:
			for param in self.parameters():
				param.requires_grad = False

	def forward(self, X):
		h_relu1 = self.slice1(X)
		h_relu2 = self.slice2(h_relu1)        
		h_relu3 = self.slice3(h_relu2)        
		h_relu4 = self.slice4(h_relu3)        
		h_relu5 = self.slice5(h_relu4)                
		out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
		return out


# --------------------------------------------------------------------------
from torchvision.models.vgg import vgg19

class TextureDescriptor(nn.Module):

	def __init__(self, device, low_level=False):
		super(TextureDescriptor, self).__init__()
		self.device = device
		self.outputs = []

		# get VGG19 feature network in evaluation mode
		self.net = vgg19(True).features.to(device)
		self.net.eval()

		# change max pooling to average pooling
		for i, x in enumerate(self.net):
			if isinstance(x, nn.MaxPool2d):
				self.net[i] = nn.AvgPool2d(kernel_size=2)

		def hook(module, input, output):
			self.outputs.append(output)

		#for i in [6, 13, 26, 39]: # with BN
		if low_level:
			for i in [4, 9]: # without BN
				self.net[i].register_forward_hook(hook)			
		else:
			for i in [4, 9, 18, 27]: # without BN
				self.net[i].register_forward_hook(hook)

		# weight proportional to num. of feature channels [Aittala 2016]
		self.weights = [1, 2, 4, 8, 8]

		# this appears to be standard for the ImageNet models in torchvision.models;
		# takes image input in [0,1] and transforms to roughly zero mean and unit stddev
		self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
		self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

	def forward(self, x):
		self.outputs = []

		# run VGG features
		x = self.net(x)
		self.outputs.append(x)

		result = []
		batch = self.outputs[0].shape[0]

		for i in range(batch):
			temp_result = []
			for j, F in enumerate(self.outputs):

				# print(j, ' shape: ', F.shape)

				F_slice = F[i,:,:,:]
				f, s1, s2 = F_slice.shape
				s = s1 * s2
				F_slice = F_slice.view((f, s))

				# Gram matrix
				G = torch.mm(F_slice, F_slice.t()) / s
				temp_result.append(G.flatten())
			temp_result = torch.cat(temp_result)

			result.append(temp_result)
		return torch.stack(result)

	def eval_CHW_tensor(self, x):
		"only takes a pytorch tensor of size B * C * H * W"
		assert len(x.shape) == 4, "input Tensor cannot be reduced to a 3D tensor"
		x = (x - self.mean) / self.std
		return self.forward(x.to(self.device))


class TDLoss(nn.Module):
	def __init__(self, GT_img, device, num_pyramid, low_level=False):
		super(TDLoss, self).__init__()
		# create texture descriptor
		self.net_td = TextureDescriptor(device, low_level=low_level) 
		# fix parameters for evaluation 
		for param in self.net_td.parameters(): 
			param.requires_grad = False 

		self.num_pyramid = num_pyramid

		self.GT_td = self.compute_td_pyramid(GT_img.to(device))


	def forward(self, img):

		# td1 = self.compute_td_pyramid(img1)
		td = self.compute_td_pyramid(img)

		tdloss = (td - self.GT_td).abs().mean() 

		return tdloss


	def compute_td_pyramid(self, img): # img: [0,1]
		"""compute texture descriptor pyramid

		Args:
			img (tensor): 4D tensor of image (NCHW)
			num_pyramid (int): pyramid level]

		Returns:
			Tensor: 2-d tensor of texture descriptor
		"""    
		# print('img type',img[0,:,0,0])
		# print('img type',img.dtype)

		# if img.dtype=='torch.uint8':

		td = self.net_td.eval_CHW_tensor(img) 
		for scale in range(self.num_pyramid):
			td_ = self.net_td.eval_CHW_tensor(nn.functional.interpolate(img, scale_factor = 1.0/(2.0**(scale+1)), mode='bilinear', align_corners=True, recompute_scale_factor=True))
			td = torch.cat([td, td_], dim=1) 
		return td


class TDLoss2(nn.Module):
	def __init__(self, device, num_pyramid, low_level=False):
		super(TDLoss2, self).__init__()
		# create texture descriptor
		self.net_td = TextureDescriptor(device, low_level=low_level) 
		# fix parameters for evaluation 
		for param in self.net_td.parameters(): 
			param.requires_grad = False 

		self.num_pyramid = num_pyramid

		# self.GT_td = self.compute_td_pyramid(GT_img.to(device))


	def forward(self, img1, img2):

		td1 = self.compute_td_pyramid(img1)
		td2 = self.compute_td_pyramid(img2)

		tdloss = (td2 - td1).abs().mean() 

		return tdloss


	def compute_td_pyramid(self, img): # img: [0,1]
		"""compute texture descriptor pyramid

		Args:
			img (tensor): 4D tensor of image (NCHW)
			num_pyramid (int): pyramid level]

		Returns:
			Tensor: 2-d tensor of texture descriptor
		"""    
		# print('img type',img[0,:,0,0])
		# print('img type',img.dtype)

		# if img.dtype=='torch.uint8':

		td = self.net_td.eval_CHW_tensor(img) 
		for scale in range(self.num_pyramid):
			td_ = self.net_td.eval_CHW_tensor(nn.functional.interpolate(img, scale_factor = 1.0/(2.0**(scale+1)), mode='bilinear', align_corners=True, recompute_scale_factor=True))
			td = torch.cat([td, td_], dim=1) 
		return td


# --------------- MG loss ------------------------------------------

from torchvision.transforms import Normalize
import torch.nn.functional as F

def normalize_vgg19(input, isGram):

	input = input/255.0
	
	if isGram:
		transform = Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.255]
		)
	else:
		transform = Normalize(
			mean=[0.48501961, 0.45795686, 0.40760392],
			std=[1./255, 1./255, 1./255]
		)
	return transform(input.cpu()).cuda()

class FeatureLoss(torch.nn.Module):

	def __init__(self, dir, w):
		super(FeatureLoss, self).__init__()

		self.net = VGG()
		self.net.load_state_dict(torch.load(dir))
		self.net.eval().cuda()

		# self.layer = ['r11','r12','r33','r43']
		self.layer = ['r11','r12','r32','r42']
		self.weights = w

	def forward(self, x):
		outputs = self.net(x, self.layer)
		# th.save(outputs, 'tmp.pt')
		# exit()
		result = []
		for i, feature in enumerate(outputs):
			result.append(feature.flatten() * self.weights[i])

		return torch.cat(result)



class VGG(nn.Module):
	def __init__(self, pool='max'):
		super(VGG, self).__init__()
		#vgg modules
		self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
		self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		if pool == 'max':
			self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
			self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
			self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
			self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
			self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
		elif pool == 'avg':
			self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
			self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
			self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
			self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
			self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

	def forward(self, x, out_keys):
		out = {}
		out['r11'] = F.relu(self.conv1_1(x))
		out['r12'] = F.relu(self.conv1_2(out['r11']))
		out['p1'] = self.pool1(out['r12'])
		out['r21'] = F.relu(self.conv2_1(out['p1']))
		out['r22'] = F.relu(self.conv2_2(out['r21']))
		out['p2'] = self.pool2(out['r22'])
		out['r31'] = F.relu(self.conv3_1(out['p2']))
		out['r32'] = F.relu(self.conv3_2(out['r31']))
		out['r33'] = F.relu(self.conv3_3(out['r32']))
		out['r34'] = F.relu(self.conv3_4(out['r33']))
		out['p3'] = self.pool3(out['r34'])
		out['r41'] = F.relu(self.conv4_1(out['p3']))
		out['r42'] = F.relu(self.conv4_2(out['r41']))
		out['r43'] = F.relu(self.conv4_3(out['r42']))
		# out['r44'] = F.relu(self.conv4_4(out['r43']))
		# out['p4'] = self.pool4(out['r44'])
		# out['r51'] = F.relu(self.conv5_1(out['p4']))
		# out['r52'] = F.relu(self.conv5_2(out['r51']))
		# out['r53'] = F.relu(self.conv5_3(out['r52']))
		# out['r54'] = F.relu(self.conv5_4(out['r53']))
		# out['p5'] = self.pool5(out['r54'])
		return [out[key] for key in out_keys]
