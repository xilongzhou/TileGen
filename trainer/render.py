import torch 
import numpy as np
# from util.util import *

eps = 1e-6

def set_param(device='cpu'):
	size = 4.0
	light_pos = torch.tensor([0.0, 0.0, 4], dtype=torch.float32).view(1, 3, 1, 1)
	light = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).view(1, 3, 1, 1) * 16 * np.pi

	return light.to(device), light_pos.to(device), size


def AdotB(a, b):
	return (a*b).sum(1, keepdim=True).clamp(min=0).expand(-1,3,-1,-1)


def norm(vec): #[B,C,W,H]
	vec = vec.div(vec.norm(2.0, 1, keepdim=True))
	return vec


def GGX(cos_h, alpha):
	c2 = cos_h**2
	a2 = alpha**2
	den = c2 * a2 + (1 - c2)
	return a2 / (np.pi * den**2 + 1e-6)

def Fresnel(cos, f0):
	return f0 + (1 - f0) * (1 - cos)**5

def Fresnel_S(cos, specular):
	sphg = torch.pow(2.0, ((-5.55473 * cos) - 6.98316) * cos)
	return specular + (1.0 - specular) * sphg

def Smith(n_dot_v, n_dot_l, alpha):
	def _G1(cos, k):
		return cos / (cos * (1.0 - k) + k)
	k = (alpha * 0.5).clamp(min=1e-6)
	return _G1(n_dot_v, k) * _G1(n_dot_l, k)

# def norm(vec): #[B,C,W,H]
# 	vec = vec.div(vec.norm(2.0, 1, keepdim=True))
# 	return vec

def getDir(pos, tex_pos):
	vec = pos - tex_pos
	return norm(vec), (vec**2).sum(1, keepdim=True)

# def AdotB(a, b):
# 	return (a*b).sum(1, keepdim=True).clamp(min=0).expand(-1,3,-1,-1)
def getTexPos(res, size, device):
	x = torch.arange(res, dtype=torch.float32)
	x = ((x + 0.5) / res - 0.5) * size

	# surface positions,
	y, x = torch.meshgrid((x, x))
	z = torch.zeros_like(x)
	pos = torch.stack((x, -y, z), 0).to(device)

	return pos

def render(maps, tex_pos, li_color, camli_pos, gamma=True, device='cuda', isMetallic=False, no_decay=False):

	assert len(li_color.shape)==4, "dim of the shape of li_color pos should be 4"
	assert len(camli_pos.shape)==4, "dim of the shape of camlight pos should be 4"
	assert len(tex_pos.shape)==4, "dim of the shape of position map should be 4"
	assert len(maps.shape)==4, "dim of the shape of feature map should be 4"
	assert camli_pos.shape[1]==3, "the 1 channel of position map should be 3"

	normal = maps[:,0:3,:,:]
	albedo = maps[:,3:6,:,:]
	rough = maps[:,6:9,:,:]
	if isMetallic:
		metallic = maps[:,9:12,:,:]

		f0 = 0.04

		# update albedo using metallic
		f0 = f0 + metallic * (albedo - f0)
		albedo = albedo * (1.0 - metallic) 

	else:
		f0 = 0.04

	v, _ = getDir(camli_pos, tex_pos)
	l, dist_l_sq = getDir(camli_pos, tex_pos)
	h = norm(l + v)
	normal = norm(normal)

	n_dot_v = AdotB(normal, v)
	n_dot_l = AdotB(normal, l)
	n_dot_h = AdotB(normal, h)
	v_dot_h = AdotB(v, h)

	# print('dist_l_sq:',dist_l_sq)
	if no_decay:
		geom = n_dot_l
	else:
		geom = n_dot_l / (dist_l_sq + eps)
	# geom = n_dot_l 

	D = GGX(n_dot_h, rough**2)

	F = Fresnel(v_dot_h, f0)

	G = Smith(n_dot_v, n_dot_l, rough**2)

	## lambert brdf
	f1 = albedo / np.pi

	## cook-torrance brdf
	f2 = D * F * G / (4 * n_dot_v * n_dot_l + eps)

	f = f1 + f2

	img = f * geom * li_color

	if gamma:
		return img.clamp(eps, 1.0)**(1/2.2)		
	else:
		return img.clamp(eps, 1.0)


def affine_img(opt, x, scale_factor, device):

	light, light_pos, size = set_params(opt, device)
	tex_pos = getTexPos(opt.res, size, device).unsqueeze(0)

	_,_,w,h = x.shape

	x = x.tile((1,1,3,3)).to(device)

	# [opt.ngf, 2] --> [opt.ngf, 2, 3]
	theta_temp = torch.zeros(1,6).to(device)
	theta_temp[:,0] = scale_factor
	theta_temp[:,4] = scale_factor
	in_theta = theta_temp.view(-1,2,3)

	grid = F.affine_grid(in_theta,x.size())
	x = F.grid_sample(x, grid)
		
	x = x[:,:,w:2*w,h:2*h]

	normal = x[:,0:3,:,:]*2-1
	albedo = x[:,3:6,:,:]**2.2
	rough = x[:,6:7,:,:].expand(-1,3,-1,-1)
	specular = torch.tensor([0.04]).expand(x.shape[0], 3, x.shape[2], x.shape[3]).to(device)

	out_maps = torch.cat((normal, albedo, rough, specular), dim=1)

	out_ren = render(out_maps, tex_pos, light, light_pos, device)

	return out_ren

#[B,c,H,W]
def height_to_normal(img_in, mode='tangent_space', normal_format='gl', use_input_alpha=False, use_alpha=False, intensity=10.0):
	"""Atomic function: Normal (https://docs.substance3d.com/sddoc/normal-172825289.html)

	Args:
		img_in (tensor): Input image.
		mode (str, optional): 'tangent space' or 'object space'. Defaults to 'tangent_space'.
		normal_format (str, optional): 'gl' or 'dx'. Defaults to 'dx'.
		use_input_alpha (bool, optional): Use input alpha. Defaults to False.
		use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
		intensity (float, optional): Normalized height map multiplier on dx, dy. Defaults to 1.0/3.0.
		max_intensity (float, optional): Maximum height map multiplier. Defaults to 3.0.

	Returns:
		Tensor: Normal image.
	"""
	# grayscale_input_check(img_in, "input height field")
	assert img_in.shape[1]==1, 'should be grayscale image'

	def roll_row(img_in, n):
		return img_in.roll(n, 2)

	def roll_col(img_in, n):
		return img_in.roll(n, 3)

	def norm(vec): #[B,C,W,H]
		vec = vec.div(vec.norm(2.0, 1, keepdim=True))
		return vec

	img_size = img_in.shape[2]
	# intensity = intensity * max_intensity * img_size / 256.0 # magic number to match sbs, check it later
	# intensity = 10 # magic number to match sbs, check it later
	# intensity = (intensity * 2.0 - 1.0) * max_intensity * img_size / 256.0 # magic number to match sbs, check it later
	dx = (roll_col(img_in, 1) - roll_col(img_in, -1))*0.5
	dy = (roll_row(img_in, 1) - roll_row(img_in, -1))*0.5
	if normal_format == 'gl':
		img_out = torch.cat((intensity*dx, -intensity*dy, torch.ones_like(dx)), 1)
	elif normal_format == 'dx':
		img_out = torch.cat((intensity*dx, intensity*dy, torch.ones_like(dx)), 1)
	else:
		img_out = torch.cat((-intensity*dx, intensity*dy, torch.ones_like(dx)), 1)
	img_out = norm(img_out)
	if mode == 'tangent_space':
		img_out = img_out / 2.0 + 0.5 #[-1,1]->[0,1]
	

	return img_out


# ## orign version
# #[B,c,H,W]
# def height_to_normal(img_in, mode='tangent_space', normal_format='gl', use_input_alpha=False, use_alpha=False, intensity=10.0):
#     """Atomic function: Normal (https://docs.substance3d.com/sddoc/normal-172825289.html)

#     Args:
#         img_in (tensor): Input image.
#         mode (str, optional): 'tangent space' or 'object space'. Defaults to 'tangent_space'.
#         normal_format (str, optional): 'gl' or 'dx'. Defaults to 'dx'.
#         use_input_alpha (bool, optional): Use input alpha. Defaults to False.
#         use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
#         intensity (float, optional): Normalized height map multiplier on dx, dy. Defaults to 1.0/3.0.
#         max_intensity (float, optional): Maximum height map multiplier. Defaults to 3.0.

#     Returns:
#         Tensor: Normal image.
#     """
#     # grayscale_input_check(img_in, "input height field")
#     assert img_in.shape[1]==1, 'should be grayscale image'

#     def roll_row(img_in, n):
#         return img_in.roll(n, 2)

#     def roll_col(img_in, n):
#         return img_in.roll(n, 3)

#     def norm(vec): #[B,C,W,H]
#         vec = vec.div(vec.norm(2.0, 1, keepdim=True))
#         return vec

#     img_size = img_in.shape[2]
#     # intensity = intensity * max_intensity * img_size / 256.0 # magic number to match sbs, check it later
#     # intensity = 10 # magic number to match sbs, check it later
#     # intensity = (intensity * 2.0 - 1.0) * max_intensity * img_size / 256.0 # magic number to match sbs, check it later
#     dx = roll_col(img_in, 1) - img_in
#     dy = roll_row(img_in, 1) - img_in
#     if normal_format == 'gl':
#         img_out = torch.cat((intensity*dx, -intensity*dy, torch.ones_like(dx)), 1)
#     elif normal_format == 'dx':
#         img_out = torch.cat((intensity*dx, intensity*dy, torch.ones_like(dx)), 1)
#     else:
#         img_out = torch.cat((-intensity*dx, intensity*dy, torch.ones_like(dx)), 1)
#     img_out = norm(img_out)
#     if mode == 'tangent_space':
#         img_out = img_out / 2.0 + 0.5 #[-1,1]->[0,1]
	

#     return img_out

if __name__ == '__main__':

	import argparse
	import os

	from PIL import Image
	import torchvision.transforms as transforms
	from torchvision import  utils

	parser = argparse.ArgumentParser()
	parser.add_argument("--in_path", type=str, help='input path') 
	parser.add_argument("--out_path", type=str, help='output path') 
	args = parser.parse_args()

	if not os.path.exists(args.out_path):
		os.makedirs(args.out_path)

	# load feature maps
	allfiles=os.listdir(args.in_path)

	for index, file in enumerate(allfiles):
		print(index)
		path = os.path.join(args.in_path, file)
		pat_pil = Image.open(path)

		toTensor = transforms.ToTensor()
		full_img = toTensor(pat_pil).cuda()

		c,h,w = full_img.shape

		H = full_img[0:1,:,0:h]
		D = full_img[:,:,h:2*h]
		R = full_img[0:1,:,2*h:3*h]
		feas = torch.cat([H, D, R], dim=0).unsqueeze(0)
		R = torch.clamp(R,min=0.03)

		light, light_pos, size = set_param('cuda')
		fake_N = height_to_normal(feas[:,0:1,:,:])
		feas = torch.cat((2*fake_N-1,feas[:,1:4,:,:],feas[:,4:5,:,:].repeat(1,3,1,1)),dim=1)
		tex_pos = getTexPos(feas.shape[2], size, 'cuda').unsqueeze(0)
		rens = render(feas, tex_pos, light, light_pos, isSpecular=False, no_decay=False) #[0,1]

		save_path = os.path.join(args.out_path, str(index)+'_Ren.png')

		utils.save_image( rens, save_path, nrow=1, normalize=False)


	# pass

