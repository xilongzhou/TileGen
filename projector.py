import argparse
from trainer.scene_trainer import Trainer
import torch 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from torch import nn
from models.scene_model import Generator

import numpy as np
import os

from PIL import Image
import torchvision.transforms as transforms

from torchvision import  utils
from trainer.render import render, set_param, getTexPos, height_to_normal
from optim_utils import TDLoss, normalize_vgg19, FeatureLoss

import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from trainer.utils import mycrop

import random
import imageio

import shutil
import subprocess

light, light_pos, size = set_param('cuda')
tex_pos = getTexPos(512, size, 'cuda').unsqueeze(0)
tex_pos_t = getTexPos(1024, size, 'cuda').unsqueeze(0)

def save_loss(loss, step, save_dir):
	plt.figure()
	plt.plot(step, loss)
	plt.savefig(save_dir+'/loss.png')
	plt.close() 

def projector(args, device, in_pats):

	down_size=args.down_size
	g_ema = Generator(args,device).to(device)
	print(args.optim, 'optim gamma: ', args.opt_gamma, 'mathch D', args.match_D)
	print("load ckpt: ", args.ckpt)
	ckpt = torch.load(os.path.join('output', args.name, 'checkpoint_eval', args.ckpt), map_location=lambda storage, loc: storage)
	g_ema.load_state_dict(ckpt["g_ema"])

	del ckpt

	if args.init=='mean':

		# style
		mean_w = g_ema.mean_latent(5000)
		if args.optim=='w' or  args.optim=='wn':
			w_opt = torch.tensor(mean_w, dtype=torch.float32, device=device, requires_grad=True)
		elif args.optim=='w+' or  args.optim=='w+n': 
			if args.starting_height_size==32:
				num_w = 9 if args.scene_size[-1]==512 else 7
			elif args.starting_height_size==4:
				num_w = 15 if args.scene_size[-1]==512 else 13
			w_opt = torch.tensor(mean_w.repeat(1,num_w,1), dtype=torch.float32, device=device, requires_grad=True)

		# noise
		noise = g_ema.make_noise()
		noise_len = len(noise)
		noise_opt=[]
		if 'n' in args.optim:
			for idx, temp in enumerate(noise):
				noise_opt.append(temp.requires_grad_(True)) 
		else:
			noise_opt = noise  

	elif args.init=='embed':

		if args.dataset=='Stone' or args.dataset=='Metal':
			embed_w = torch.load(args.embed_path)['w_plus']
			embed_noise = torch.load(args.embed_path)['noise']
			print('...................loading embed sucessfully.................')
		else:
			embed_w, embed_noise = embed(args, device, in_pats=in_pats)

		w_opt = torch.tensor(embed_w, dtype=torch.float32, device=device, requires_grad=True)

		noise_len = len(embed_noise)
		noise_opt=[]
		if 'n' in args.optim:
			for idx, temp in enumerate(embed_noise):
				noise_opt.append(temp.requires_grad_(True)) 
		else:
			noise_opt = embed_noise 
	
	elif args.init=='rand':

		# style
		mean_w = g_ema.mean_latent(1)
		if args.optim=='w' or  args.optim=='wn':
			w_opt = torch.tensor(mean_w, dtype=torch.float32, device=device, requires_grad=True)
		elif args.optim=='w+' or  args.optim=='w+n': 
			if args.starting_height_size==32:
				num_w = 9 if args.scene_size[-1]==512 else 7
			elif args.starting_height_size==4:
				num_w = 15 if args.scene_size[-1]==512 else 13
			w_opt = torch.tensor(mean_w.repeat(1,num_w,1), dtype=torch.float32, device=device, requires_grad=True)

		# noise
		noise = g_ema.make_noise()
		noise_len = len(noise)
		noise_opt=[]
		if 'n' in args.optim:
			for idx, temp in enumerate(noise):
				noise_opt.append(temp.requires_grad_(True)) 
		else:
			noise_opt = noise  

	if args.opt_gamma:
		gamma_opt = torch.tensor([1.0]).cuda().requires_grad_(True)

	# if args.opt_scale:
	height_opt = torch.tensor([args.H_scale]).cuda().requires_grad_(True)
	light_opt = torch.tensor([1.0]).cuda().requires_grad_(True)

	if args.optim=='wn' or args.optim=='w+n':
		if args.opt_gamma:
			optimizer = torch.optim.Adam(list(noise_opt)+[w_opt]+[gamma_opt], betas=(0.9, 0.999), lr=args.lr)
		elif args.opt_scale:
			optimizer = torch.optim.Adam(list(noise_opt)+[w_opt]+[height_opt]+[light_opt], betas=(0.9, 0.999), lr=args.lr)
		else:
			optimizer = torch.optim.Adam(list(noise_opt)+[w_opt], betas=(0.9, 0.999), lr=args.lr)
	elif args.optim=='n':
		optimizer = torch.optim.Adam(list(noise_opt), betas=(0.9, 0.999), lr=args.lr)
	elif args.optim=='w' or args.optim=='w+':
		if args.opt_gamma:
			optimizer = torch.optim.Adam([w_opt]+[gamma_opt], betas=(0.9, 0.999), lr=args.lr)
		else:
			optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=args.lr)

	scheduler = StepLR(optimizer, step_size=500, gamma=0.5)

	# load target image
	target_pil = Image.open(args.target).convert('RGB')
	target_uint8 = np.array(target_pil, dtype=np.uint8)
	gt_image_255 = torch.from_numpy(target_uint8).permute(2,0,1).unsqueeze(0).cuda()
	gt_image = gt_image_255/255.0

	gt_img16 = F.interpolate(gt_image, size=(down_size,down_size), mode='bilinear', align_corners=True)
	# gt_img256 = F.interpolate(gt_image, size=(256,256), mode='bilinear', align_corners=True)

	print(gt_image.shape)
	criterionTD = TDLoss(gt_image.clone(), device, 2)
	criterionL1 = torch.nn.L1Loss()
	criterionL2 = torch.nn.MSELoss()

	if args.loss=='MG':
		MG_loss = FeatureLoss('vgg_conv.pt', [0.125, 0.125, 0.125, 0.125])
		for p in MG_loss.parameters():
			p.requires_grad = False
		gt_fea = MG_loss(normalize_vgg19(gt_image_255.clone(), False))

	if not args.no_cond:
		utils.save_image( in_pats.unsqueeze(0), os.path.join(args.out_path, 'inpat.png'), normalize=True, range=(-1,1))
	utils.save_image( gt_image, os.path.join(args.out_path,'target.png'), normalize=False)

	if args.pad_optim:
		padding=5
		tex_pos_pad = getTexPos(512+padding*2, size, 'cuda').unsqueeze(0)

	isMetallic = args.isMetallic

	TotalLoss_log=[]
	Plot_iter=[]

	shift_optim=True if args.shift_optim else False

	for index in range(args.steps + 1):

		if args.inter_shift:
			shift_optim=True if index%2==1 else False

		if index==args.steps:
			shift_optim=False

		if args.optim=='n':
			output = g_ema(in_pats.unsqueeze(0), noise=noise_opt)['image']  
		elif args.optim=='wn' or args.optim=='w':
			output = g_ema(in_pats.unsqueeze(0), noise=noise_opt, styles=w_opt, input_type='w')['image']
		elif args.optim=='w+n' or args.optim=='w+':
			output = g_ema(in_pats.unsqueeze(0), noise=noise_opt, styles=w_opt, input_type='w+')['image']

		if index!=args.steps+1 and shift_optim:
			rand1 = [random.randint(1,511), random.randint(1,511)]
			output = mycrop(output, in_pats.shape[-1], rand0=rand1)

		if args.shift_optim_L1:
			rand1 = [random.randint(1,511), random.randint(1,511)]
			output_s = mycrop(output, in_pats.shape[-1], rand0=rand1)

		if args.pad_optim:
			output = F.pad(output, (padding,padding,padding,padding), mode ='circular')

		if args.opt_scale:

			fea = output*0.5+0.5
			R = fea[:,4:5,:,:]**gamma_opt if args.opt_gamma else fea[:,4:5,:,:]

			if args.dataset=='Metal':
				R = torch.clamp(R, min=args.cut_rough)

			fake_N = height_to_normal(fea[:,0:1,:,:], intensity=height_opt)

			if isMetallic:
				M = fea[:,5:6,:,:]
				ren_fea = torch.cat((2*fake_N-1,fea[:,1:4,:,:],R.repeat(1,3,1,1),M.repeat(1,3,1,1)),dim=1)
			else:
				ren_fea = torch.cat((2*fake_N-1,fea[:,1:4,:,:],R.repeat(1,3,1,1)),dim=1)

			if args.pad_optim:
				out_rens = render(ren_fea, tex_pos_pad, light*light_opt, light_pos, isMetallic=isMetallic, no_decay=False) #[0,1]            
			else:                
				out_rens = render(ren_fea, tex_pos, light*light_opt, light_pos, isMetallic=isMetallic, no_decay=False) #[0,1]   
			

			if args.shift_optim_L1:

				fea_s = output_s*0.5+0.5
				R_s = fea_s[:,4:5,:,:]**gamma_opt if args.opt_gamma else fea_s[:,4:5,:,:]
				fake_N_s = height_to_normal(fea_s[:,0:1,:,:], intensity=height_opt)

				if isMetallic:
					M_s = fea_s[:,5:6,:,:]
					ren_fea_s = torch.cat((2*fake_N_s-1,fea_s[:,1:4,:,:],R_s.repeat(1,3,1,1),M_s.repeat(1,3,1,1)),dim=1)
				else:
					ren_fea_s = torch.cat((2*fake_N_s-1,fea_s[:,1:4,:,:],R_s.repeat(1,3,1,1)),dim=1)

				if args.pad_optim:
					out_rens_s = render(ren_fea_s, tex_pos_pad, light*light_opt, light_pos, isMetallic=isMetallic, no_decay=False) #[0,1]            
				else:                
					out_rens_s = render(ren_fea_s, tex_pos, light*light_opt, light_pos, isMetallic=isMetallic, no_decay=False) #[0,1]  

		else:
			fea = output*0.5+0.5
			R = fea[:,4:5,:,:]**gamma_opt if args.opt_gamma else fea[:,4:5,:,:]

			# 0.05 R
			R = torch.clamp(R, min=0.05)

			fake_N = height_to_normal(fea[:,0:1,:,:])

			if isMetallic:
				M = fea[:,5:6,:,:]
				ren_fea = torch.cat((2*fake_N-1,fea[:,1:4,:,:],R.repeat(1,3,1,1),M.repeat(1,3,1,1)),dim=1)
			else:
				ren_fea = torch.cat((2*fake_N-1,fea[:,1:4,:,:],R.repeat(1,3,1,1)),dim=1)

			if args.pad_optim:
				out_rens = render(ren_fea, tex_pos_pad, light, light_pos, isMetallic=isMetallic, no_decay=False) #[0,1] 
			else:           
				out_rens = render(ren_fea, tex_pos, light, light_pos, isMetallic=isMetallic, no_decay=False) #[0,1]            

		if args.shift_optim_L1:
			out_rens16 = F.interpolate(out_rens_s, size=(down_size,down_size), mode='bilinear', align_corners=True)
		else:
			out_rens16 = F.interpolate(out_rens, size=(down_size,down_size), mode='bilinear', align_corners=True)

		TD_loss = criterionTD(out_rens)
		pixel_loss = args.weight_L1*criterionL1(gt_img16, out_rens16)
		loss =  TD_loss + pixel_loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()

		if args.pad_optim:
			fake_N = fake_N[:, :, padding:512+padding, padding:512+padding]
			fea = fea[:, :, padding:512+padding, padding:512+padding]
			R = R[:, :, padding:512+padding, padding:512+padding]
			out_rens = out_rens[:, :, padding:512+padding, padding:512+padding]

		temp_loss = loss.detach().cpu().numpy()

		if index%100==0:
			TotalLoss_log.append(temp_loss)
			Plot_iter.append(index)

		# print(loss.detach().cpu().numpy())

		if index% int(args.steps/2)==0:            
			out_lr = optimizer.param_groups[0]['lr']
			print(f'index {index}, loss {temp_loss:.5f}, TD_loss {TD_loss:.5f}, pixel_loss {pixel_loss:.5f}, height_opt: {height_opt.item():2f}, light_opt: {light_opt.item():2f}, lr: {out_lr:.5f}') 
			print(shift_optim)
			if isMetallic:
				vis_fea = torch.cat((fea[:,0:1,:,:].repeat(1,3,1,1),fea[:,1:4,:,:]**(1/2.2),R.repeat(1,3,1,1),fea[:,5:6,:,:].repeat(1,3,1,1)),dim=-1)
			else:
				vis_fea = torch.cat((fea[:,0:1,:,:].repeat(1,3,1,1),fea[:,1:4,:,:]**(1/2.2),R.repeat(1,3,1,1)),dim=-1)

			utils.save_image( out_rens, os.path.join(args.out_path, f'{index}_fakerens.png'), normalize=False)
			utils.save_image( vis_fea, os.path.join(args.out_path, f'{index}_fakefea.png'), normalize=False)                

	del pixel_loss
	del criterionTD
	del out_rens16
	del gt_img16

	save_loss(TotalLoss_log, Plot_iter, args.out_path )

	# save each feature maps
	utils.save_image( fake_N, os.path.join(args.out_path, 'opt_N.png'), normalize=False)
	utils.save_image( fea[:,0:1,:,:], os.path.join(args.out_path, 'opt_H.png'), normalize=False)
	utils.save_image( fea[:,1:4,:,:]**(1/2.2), os.path.join(args.out_path, 'opt_D.png'), normalize=False)
	utils.save_image( R, os.path.join(args.out_path, 'opt_R.png'), normalize=False)

	if isMetallic:
		utils.save_image( fea[:,5:6,:,:], os.path.join(args.out_path, 'opt_M.png'), normalize=False)
		Combined_fea = torch.cat([torch.cat([fea[:,1:4,:,:]**(1/2.2),fea[:,0:1,:,:].repeat(1,3,1,1)],dim=-1), torch.cat([R.repeat(1,3,1,1),fea[:,5:6,:,:].repeat(1,3,1,1)],dim=-1)], dim=-2)
		utils.save_image( Combined_fea, os.path.join(args.out_path, 'opt_fea.png'), normalize=False )
	else:
		Combined_fea = torch.cat([torch.cat([fea[:,1:4,:,:]**(1/2.2),fea[:,0:1,:,:].repeat(1,3,1,1)],dim=-1), torch.cat([R.repeat(1,3,1,1),fake_N],dim=-1)], dim=-2)
		utils.save_image( Combined_fea, os.path.join(args.out_path, 'opt_fea.png'), normalize=False )        

	H_tile = torch.tile(fea[:,0:1,:,:],(1,1,2,2))
	D_tile = torch.tile(fea[:,1:4,:,:],(1,1,2,2))
	R_tile = torch.tile(R,(1,1,2,2))
	if args.opt_scale:
		N_tile = height_to_normal(H_tile, intensity=height_opt)
		ren_fea_t = torch.cat((2*N_tile-1,D_tile,R_tile),dim=1)
		out_rens_t = render(ren_fea_t, tex_pos_t, light*light_opt, light_pos, isMetallic=False, no_decay=False) #[0,1] 
	else:
		N_tile = height_to_normal(H_tile)
		ren_fea_t = torch.cat((2*N_tile-1,D_tile,R_tile),dim=1)
		out_rens_t = render(ren_fea_t, tex_pos_t, light, light_pos, isMetallic=False, no_decay=False) #[0,1]  

	if isMetallic:
		M_tile = torch.tile(fea[:,5:6,:,:],(1,1,2,2))
		vis_fea = torch.cat((H_tile.repeat(1,3,1,1),D_tile**(1/2.2),R_tile.repeat(1,3,1,1),M_tile.repeat(1,3,1,1)),dim=-1)
		utils.save_image( M_tile, os.path.join(args.out_path, 'opt_M_t.png'), normalize=False)
	else:
		vis_fea = torch.cat((H_tile.repeat(1,3,1,1),D_tile**(1/2.2),R_tile.repeat(1,3,1,1)),dim=-1)

	utils.save_image( vis_fea, os.path.join(args.out_path, 'opt_fakefea_t.png'), normalize=False)
	utils.save_image( N_tile, os.path.join(args.out_path, 'opt_N_t.png'), normalize=False)
	utils.save_image( H_tile, os.path.join(args.out_path, 'opt_H_t.png'), normalize=False)
	utils.save_image( D_tile**(1/2.2), os.path.join(args.out_path, 'opt_D_t.png'), normalize=False)
	utils.save_image( R_tile, os.path.join(args.out_path, 'opt_R_t.png'), normalize=False)
	utils.save_image( out_rens_t, os.path.join(args.out_path, 'opt_Ren_t.png'), normalize=False)

	del gt_image
	del optimizer
	del noise_opt
	del w_opt
	del loss
	del g_ema
	del in_pats
	del TotalLoss_log
	del Plot_iter

	torch.cuda.empty_cache()

	if args.opt_scale:
		torch.save({'height': height_opt, 'light': light_opt}, os.path.join(args.out_path, 'scale.pt'))

def embed(args, device, in_pats):

	down_size=args.down_size
	g_ema = Generator(args,device).to(device)
	print(args.optim, 'optim gamma: ', args.opt_gamma, 'mathch D', args.match_D)
	print("load ckpt: ", args.ckpt)
	ckpt = torch.load(os.path.join('output', args.name, 'checkpoint_eval', args.ckpt), map_location=lambda storage, loc: storage)
	g_ema.load_state_dict(ckpt["g_ema"])
	mean_w = g_ema.mean_latent(5000)

	del ckpt

	# latent (nocond_z)
	if args.optim=='w' or  args.optim=='wn':
		w_opt = torch.tensor(mean_w, dtype=torch.float32, device=device, requires_grad=True)
	elif args.optim=='w+' or  args.optim=='w+n': 
		if args.starting_height_size==32:
			num_w = 9 if args.scene_size[-1]==512 else 7
		elif args.starting_height_size==4:
			num_w = 15 if args.scene_size[-1]==512 else 13
		w_opt = torch.tensor(mean_w.repeat(1,num_w,1), dtype=torch.float32, device=device, requires_grad=True)
	print(w_opt.shape)

	# noise
	noise = g_ema.make_noise()
	noise_len = len(noise)
	noise_opt=[]
	if 'n' in args.optim:
		for idx, temp in enumerate(noise):
			noise_opt.append(temp.requires_grad_(True))            
	else:
		noise_opt = noise       

	if args.optim=='wn' or args.optim=='w+n':
		optimizer = torch.optim.Adam(list(noise_opt)+[w_opt], betas=(0.9, 0.999), lr=0.02)
	elif args.optim=='n':
		optimizer = torch.optim.Adam(list(noise_opt), betas=(0.9, 0.999), lr=0.02)
	elif args.optim=='w' or args.optim=='w+':
		optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=0.02)

	criterionL1 = torch.nn.L1Loss()
	criterionL2 = torch.nn.MSELoss()

	isMetallic = args.isMetallic

	gt_N = torch.tensor([0.5,0.5,1], device='cuda').unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,512,512)
	if isMetallic:
		gt_tempD = torch.tensor([0.5], device='cuda').unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1,3,512,512)
		gt_tempR = torch.tensor([0.3], device='cuda').unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1,3,512,512)
		gt_tempM = torch.tensor([0.9], device='cuda').unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1,3,512,512)
		gt_temp = torch.cat([gt_tempD, gt_tempR, gt_tempM], dim=-1)
	else:
		gt_temp = torch.tensor([0.5], device='cuda').unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1,3,512,1024)

	gt_fea = torch.cat([gt_N, gt_temp], dim=-1)


	for index in range(1001):

		if args.inter_shift:
			args.shift_optim=True if index%2==0 else False

		if args.optim=='n':
			fea = g_ema(in_pats.unsqueeze(0), noise=noise_opt)['image']  
		elif args.optim=='wn' or args.optim=='w':
			fea = g_ema(in_pats.unsqueeze(0), noise=noise_opt, styles=w_opt, input_type='w')['image']
		elif args.optim=='w+n' or args.optim=='w+':
			fea = g_ema(in_pats.unsqueeze(0), noise=noise_opt, styles=w_opt, input_type='w+')['image']

		fea = fea*0.5+0.5

		fake_N = height_to_normal(fea[:,0:1,:,:], intensity=args.H_scale)

		if isMetallic:
			fea = torch.cat((fake_N,fea[:,1:4,:,:],fea[:,4:5,:,:].repeat(1,3,1,1),fea[:,5:6,:,:].repeat(1,3,1,1)),dim=-1)
		else:
			fea = torch.cat((fake_N,fea[:,1:4,:,:],fea[:,4:5,:,:].repeat(1,3,1,1)),dim=-1)

		pixel_loss = criterionL2(fea, gt_fea)           

		optimizer.zero_grad()
		pixel_loss.backward()
		optimizer.step()

		if index%1000==0:

			print(f'embed pixel_loss {pixel_loss:.5f}')

			utils.save_image( fea, os.path.join(args.out_path, f'{index}_embedfea.png'), normalize=False)                

	torch.save({'noise':noise_opt, 'w_plus':w_opt}, 'embed.pt')

	del optimizer
	del pixel_loss
	del g_ema

	torch.cuda.empty_cache()

	return w_opt,noise_opt



def sample(args, device, in_pats, idx):

	N_perimg=1

	nrow = 4 if N_perimg==16 else 1

	output_interfea = False # save intermediate featuer or not

	isMetallic = args.isMetallic

	if args.no_cond:
		tmp = torch.ones((1,1,512,512))

		N=100

		out_path = os.path.join('output', args.name, str(args.mode), args.savename)

		if not os.path.exists(out_path):
			os.makedirs(out_path)   
			
		# for each pattern, sample 10 examples
		for k in range(N):

			g_ema = Generator(args,device).to(device)

			print("load ckpt: ", args.ckpt)
			ckpt = torch.load(os.path.join('output', args.name, 'checkpoint_eval', args.ckpt), map_location=lambda storage, loc: storage)

			g_ema.load_state_dict(ckpt["g_ema"])

			D = []
			H = []
			N = []
			R = []
			M = []
			Ren = []
			Ren_tile = []
			jitter = torch.randn([1, 10], device=device)*0.1 if args.add_jitter else None


			if not args.nocond_z:
				fix_z = torch.randn( 1, args.style_dim, device=device)
				if output_interfea:
					output, inter_feature = g_ema(tmp, styles=[fix_z], input_type='z', jitter=jitter, out_inter=True) 
					output = output['image'] 

				else:
					output = g_ema(tmp, styles=[fix_z], input_type='z', jitter=jitter)['image']  
			else:
				if output_interfea:
					output, inter_feature = g_ema(tmp, jitter=jitter, out_inter=True)
					output = output['image'] 
				else:
					output = g_ema(tmp, jitter=jitter)['image']  

			ren_fea = output*0.5+0.5 # [-1,1] --> [0,1]

			# save feature maps
			H.append(ren_fea[:,0:1,:,:].repeat(1,3,1,1))
			D.append(ren_fea[:,1:4,:,:]**(1/2.2))
			R.append(ren_fea[:,4:5,:,:].repeat(1,3,1,1))
			if isMetallic:
				M.append(ren_fea[:,5:6,:,:].repeat(1,3,1,1))

			light, light_pos, size = set_param('cuda')
			fake_N = height_to_normal(ren_fea[:,0:1,:,:], intensity=args.H_scale)
			if isMetallic:
				ren_fea = torch.cat((2*fake_N-1,ren_fea[:,1:4,:,:],ren_fea[:,4:5,:,:].repeat(1,3,1,1), ren_fea[:,5:6,:,:].repeat(1,3,1,1)),dim=1)
			else:
				ren_fea = torch.cat((2*fake_N-1,ren_fea[:,1:4,:,:],ren_fea[:,4:5,:,:].repeat(1,3,1,1)),dim=1)
			tex_pos = getTexPos(ren_fea.shape[2], size, 'cuda').unsqueeze(0)
			rens = render(ren_fea, tex_pos, light, light_pos, isMetallic=isMetallic, no_decay=False) #[0,1]
			Ren.append(rens)
			N.append(fake_N)


			if output_interfea:
				for fea in range(len(inter_feature)):
					fea_map = inter_feature[fea]

					save_path = os.path.join(out_path, str(k) + '_' + str(fea) +'_interfea0.png')
					utils.save_image( fea_map[:,0:1,...], save_path, nrow=nrow, normalize=True, range=(-1,1) )

					save_path = os.path.join(out_path,  str(k) + '_' + str(fea) +'_interfea1.png')
					utils.save_image( fea_map[:,2:3,...], save_path, nrow=nrow, normalize=True, range=(-1,1) )

					save_path = os.path.join(out_path,  str(k) + '_' + str(fea) +'_interfea2.png')
					utils.save_image( fea_map[:,4:5,...], save_path, nrow=nrow, normalize=True, range=(-1,1) )




			H = torch.cat(H, dim=0)
			D = torch.cat(D, dim=0)
			R = torch.cat(R, dim=0)
			N = torch.cat(N, dim=0)


			# H_tile = torch.tile(H,(1,1,2,2))
			# D_tile = torch.tile(D,(1,1,2,2))
			# R_tile = torch.tile(R,(1,1,2,2))

			Ren = torch.cat(Ren, dim=0)
			# Ren_tile = torch.cat(Ren_tile, dim=0)

			# print(name, x.shape)
			save_path = os.path.join(out_path, str(k) +'_H.png')
			utils.save_image( H, save_path, nrow=nrow, normalize=False)

			# save_tile_path = os.path.join(out_path, save_idx+'_Htile.png')
			# utils.save_image( H_tile, save_tile_path, nrow=4, normalize=True, range=(-1,1) )

			save_path = os.path.join(out_path, str(k) + '_Ren.png')
			utils.save_image( Ren, save_path, nrow=nrow, normalize=False)            
			# save_path = os.path.join(out_path, save_idx+'_Rentile.png')
			# utils.save_image( Ren_tile, save_path, nrow=4, normalize=False)

			save_path = os.path.join(out_path, str(k) + '_D.png')
			utils.save_image( D, save_path, nrow=nrow, normalize=False)
			# save_path = os.path.join(out_path, save_idx+'_Dtile.png')
			# utils.save_image( D_tile, save_path, nrow=4, normalize=False)

			save_path = os.path.join(out_path, str(k) + '_R.png')
			utils.save_image( R, save_path, nrow=nrow, normalize=False,)
			# save_path = os.path.join(out_path, save_idx+'_Rtile.png')
			# utils.save_image( R_tile, save_path, nrow=4, normalize=True, range=(-1,1) )

			save_path = os.path.join(out_path, str(k) + '_N.png')
			utils.save_image( N, save_path, nrow=nrow, normalize=False )


			if isMetallic:
				M = torch.cat(M, dim=0)
				save_path = os.path.join(out_path, str(k) + '_M.png')
				utils.save_image( M, save_path, nrow=nrow, normalize=False )

				Combined_fea = torch.cat([torch.cat([D,H],dim=-1), torch.cat([R,M],dim=-1)], dim=-2)

				save_path = os.path.join(out_path, str(k) + '_fea.png')
				utils.save_image( Combined_fea, save_path, nrow=nrow, normalize=False )
			else:

				Combined_fea = torch.cat([torch.cat([D,H],dim=-1), torch.cat([R,N],dim=-1)], dim=-2)
				save_path = os.path.join(out_path, str(k) + '_fea.png')
				utils.save_image( Combined_fea, save_path, nrow=nrow, normalize=False )


	else:

		N = 1

		for n in range(N):
			out_path = os.path.join('output', args.name, str(args.mode), args.savename, str(n))

			if not os.path.exists(out_path):
				os.makedirs(out_path)   
				
			for index in range(in_pats.shape[0]):

				   
				save_idx = str(index)

				in_pat=in_pats[index,...]

				utils.save_image( in_pat.unsqueeze(0), os.path.join(out_path, save_idx+'inpat.png'), normalize=True, range=(-1,1))

				# for each pattern, sample 10 examples
				for k in range(4):

					g_ema = Generator(args,device).to(device)

					print("load ckpt: ", args.ckpt)
					ckpt = torch.load(os.path.join('output', args.name, 'checkpoint_eval', args.ckpt), map_location=lambda storage, loc: storage)

					g_ema.load_state_dict(ckpt["g_ema"])

					D = []
					H = []
					N = []
					R = []
					M = []
					Ren = []
					Ren_tile = []
					jitter = torch.randn([1, 10], device=device)*0.1 if args.add_jitter else None


					if not args.nocond_z:
						fix_z = torch.randn( 1, args.style_dim, device=device)
						if output_interfea:
							output, inter_feature = g_ema(in_pat.unsqueeze(0), styles=[fix_z], input_type='z', jitter=jitter, out_inter=True) 
							output = output['image'] 

						else:
							output = g_ema(in_pat.unsqueeze(0), styles=[fix_z], input_type='z', jitter=jitter)['image']  
					else:
						if output_interfea:
							output, inter_feature = g_ema(in_pat.unsqueeze(0), jitter=jitter, out_inter=True)
							output = output['image'] 
						else:
							output = g_ema(in_pat.unsqueeze(0), jitter=jitter)['image']  
					ren_fea = output*0.5+0.5

					# save feature maps
					H.append(ren_fea[:,0:1,:,:].repeat(1,3,1,1))
					D.append(ren_fea[:,1:4,:,:]**(1/2.2))
					R.append(ren_fea[:,4:5,:,:].repeat(1,3,1,1))
					if isMetallic:
						M.append(ren_fea[:,5:6,:,:].repeat(1,3,1,1))

					light, light_pos, size = set_param('cuda')
					fake_N = height_to_normal(ren_fea[:,0:1,:,:], intensity=args.H_scale)
					if isMetallic:
						ren_fea = torch.cat((2*fake_N-1,ren_fea[:,1:4,:,:],ren_fea[:,4:5,:,:].repeat(1,3,1,1), ren_fea[:,5:6,:,:].repeat(1,3,1,1)),dim=1)
					else:
						ren_fea = torch.cat((2*fake_N-1,ren_fea[:,1:4,:,:],ren_fea[:,4:5,:,:].repeat(1,3,1,1)),dim=1)
					tex_pos = getTexPos(ren_fea.shape[2], size, 'cuda').unsqueeze(0)
					rens = render(ren_fea, tex_pos, light, light_pos, isMetallic=isMetallic, no_decay=False) #[0,1]
					Ren.append(rens)
					N.append(fake_N)


					if output_interfea:
						for fea in range(len(inter_feature)):
							fea_map = inter_feature[fea]

							save_path = os.path.join(out_path, save_idx+ '_' + str(k) + '_' + str(fea) +'_interfea0.png')
							utils.save_image( fea_map[:,0:1,...], save_path, nrow=nrow, normalize=True, range=(-1,1) )

							save_path = os.path.join(out_path,  save_idx+ '_' + str(k) + '_' + str(fea) +'_interfea1.png')
							utils.save_image( fea_map[:,2:3,...], save_path, nrow=nrow, normalize=True, range=(-1,1) )

							save_path = os.path.join(out_path,  save_idx+ '_' + str(k) + '_' + str(fea) +'_interfea2.png')
							utils.save_image( fea_map[:,4:5,...], save_path, nrow=nrow, normalize=True, range=(-1,1) )




					H = torch.cat(H, dim=0)
					D = torch.cat(D, dim=0)
					R = torch.cat(R, dim=0)
					N = torch.cat(N, dim=0)


					# H_tile = torch.tile(H,(1,1,2,2))
					# D_tile = torch.tile(D,(1,1,2,2))
					# R_tile = torch.tile(R,(1,1,2,2))

					Ren = torch.cat(Ren, dim=0)
					# Ren_tile = torch.cat(Ren_tile, dim=0)

					# print(name, x.shape)
					save_path = os.path.join(out_path, save_idx+ '_' + str(k) +'_H.png')
					utils.save_image( H, save_path, nrow=nrow, normalize=False )

					# save_tile_path = os.path.join(out_path, save_idx+'_Htile.png')
					# utils.save_image( H_tile, save_tile_path, nrow=4, normalize=True, range=(-1,1) )

					save_path = os.path.join(out_path, save_idx+ '_' + str(k) + '_Ren.png')
					utils.save_image( Ren, save_path, nrow=nrow, normalize=False)            
					# save_path = os.path.join(out_path, save_idx+'_Rentile.png')
					# utils.save_image( Ren_tile, save_path, nrow=4, normalize=False)

					save_path = os.path.join(out_path, save_idx+ '_' + str(k) + '_D.png')
					utils.save_image( D, save_path, nrow=nrow, normalize=False)
					# save_path = os.path.join(out_path, save_idx+'_Dtile.png')
					# utils.save_image( D_tile, save_path, nrow=4, normalize=False)

					save_path = os.path.join(out_path, save_idx+ '_' + str(k) + '_R.png')
					utils.save_image( R, save_path, nrow=nrow, normalize=False )
					# save_path = os.path.join(out_path, save_idx+'_Rtile.png')
					# utils.save_image( R_tile, save_path, nrow=4, normalize=True, range=(-1,1) )

					save_path = os.path.join(out_path, save_idx+ '_' + str(k) + '_N.png')
					utils.save_image( N, save_path, nrow=nrow, normalize=False)


					if isMetallic:
						M = torch.cat(M, dim=0)
						save_path = os.path.join(out_path, save_idx+ '_' + str(k) + '_M.png')
						utils.save_image( M, save_path, nrow=nrow, normalize=False )

						Combined_fea = torch.cat([torch.cat([D,H],dim=-1), torch.cat([R,M],dim=-1)], dim=-2)

						save_path = os.path.join(out_path, save_idx+ '_' + str(k) + '_fea.png')
						utils.save_image( Combined_fea, save_path, nrow=nrow, normalize=False )
					else:

						Combined_fea = torch.cat([torch.cat([D,H],dim=-1), torch.cat([R,N],dim=-1)], dim=-2)
						save_path = os.path.join(out_path, save_idx+ '_' + str(k) + '_fea.png')
						utils.save_image( Combined_fea, save_path, nrow=nrow, normalize=False )


def interpolate(args, device, in_pats, idx):

	N = 10

	isMetallic = args.isMetallic

	for n in range(N):
		out_path = os.path.join('output', args.name, str(args.mode), args.savename, str(n))

		if not os.path.exists(out_path):
			os.makedirs(out_path)   
			
		for index in range(in_pats.shape[0]):

			save_idx = str(index)

			in_pat=in_pats[index,...]

			if not args.no_cond:
				utils.save_image( in_pat.unsqueeze(0), os.path.join(out_path, save_idx+'inpat.png'), normalize=True, range=(-1,1))

			g_ema = Generator(args,device).to(device)

			print("load ckpt: ", args.ckpt)
			ckpt = torch.load(os.path.join('output', args.name, 'checkpoint_eval', args.ckpt), map_location=lambda storage, loc: storage)

			g_ema.load_state_dict(ckpt["g_ema"])

			D = []
			H = []
			R = []
			M = []
			Ren = []
			Ren_tile = []

			fix_z1 = torch.randn( 1, args.style_dim, device=device)
			fix_z2 = torch.randn( 1, args.style_dim, device=device)

			if not args.inter_z:

				w1 = g_ema.style(fix_z1)
				w2 = g_ema.style(fix_z2)

				if args.starting_height_size==32:
					num_w = 9 if args.scene_size[-1]==512 else 7
				elif args.starting_height_size==4:
					num_w = 15 if args.scene_size[-1]==512 else 13

				print(w1.shape)
				w_plus1 = torch.tensor(w1.repeat(1,num_w,1), dtype=torch.float32, device=device)
				w_plus2 = torch.tensor(w2.repeat(1,num_w,1), dtype=torch.float32, device=device)
				print(w_plus1.shape)

			noise_fix = g_ema.make_noise()

			if args.video:
				N_inter = 50
				w_inter=0
				gif=[]
				for j in range(N_inter+1):

					# if 
					if args.inter_z:
						fix_z3 = fix_z1*w_inter + fix_z2*(1-w_inter)
						w3 = g_ema.style(fix_z3)

						if args.starting_height_size==32:
							num_w = 9 if args.scene_size[-1]==512 else 7
						elif args.starting_height_size==4:
							num_w = 15 if args.scene_size[-1]==512 else 13

						w_plus3 = torch.tensor(w3.repeat(1,num_w,1), dtype=torch.float32, device=device)
					else:
						w_plus3 = w_plus1*w_inter + w_plus2*(1-w_inter)

					output = g_ema(in_pat.unsqueeze(0), styles=w_plus3, noise=noise_fix, input_type='w+')['image']

					# save feature maps
					H.append(output[:,0:1,:,:].repeat(1,3,1,1))
					D.append((0.5*output[:,1:4,:,:]+0.5)**(1/2.2))
					R.append(output[:,4:5,:,:].repeat(1,3,1,1))
					if isMetallic:
						M.append(output[:,5:6,:,:].repeat(1,3,1,1))

					ren_fea = output*0.5+0.5
					light, light_pos, size = set_param('cuda')
					fake_N = height_to_normal(ren_fea[:,0:1,:,:], intensity=args.H_scale)
					if isMetallic:
						ren_fea = torch.cat((2*fake_N-1,ren_fea[:,1:4,:,:],ren_fea[:,4:5,:,:].repeat(1,3,1,1),ren_fea[:,5:6,:,:].repeat(1,3,1,1)),dim=1)
					else:
						ren_fea = torch.cat((2*fake_N-1,ren_fea[:,1:4,:,:],ren_fea[:,4:5,:,:].repeat(1,3,1,1)),dim=1)
					tex_pos = getTexPos(ren_fea.shape[2], size, 'cuda').unsqueeze(0)
					rens = render(ren_fea, tex_pos, light, light_pos, isMetallic=isMetallic, no_decay=False).squeeze(0).permute(1,2,0) #[0,1]
					Ren.append(rens)

					# ren_fea = output*0.5+0.5
					# light, light_pos, size = set_param('cuda')
					# fake_N = height_to_normal(ren_fea[:,0:1,:,:], intensity=1)
					# ren_fea = torch.cat((2*torch.tile(fake_N,(1,1,2,2))-1,torch.tile(ren_fea[:,1:4,:,:],(1,1,2,2)), torch.tile(ren_fea[:,4:5,:,:].repeat(1,3,1,1),(1,1,2,2))),dim=1)
					# tex_pos = getTexPos(ren_fea.shape[2], size, 'cuda').unsqueeze(0)

					# print(ren_fea.shape)
					# print(tex_pos.shape)
					# rens_tile = render(ren_fea, tex_pos, light, light_pos, isSpecular=False, no_decay=False) #[0,1]
					# Ren_tile.append(rens_tile)

					w_inter += 1/N_inter

					# print(rens.shape)
					gif_numpy = rens.cpu().numpy()*255.0
					gif_numpy = np.clip(gif_numpy, 0, 255)
					gif_numpy = gif_numpy.astype(np.uint8)
					gif.append(gif_numpy)


				save_gif_path = os.path.join(out_path, save_idx+'.gif')
				imageio.mimwrite(save_gif_path, gif, format='GIF', fps=18)
				print('save the gif',index)

			else:
				N_inter = 5
				w_inter=0
				gif=[]
				for j in range(N_inter+1):

					if args.inter_z:
						fix_z3 = fix_z1*w_inter + fix_z2*(1-w_inter)
						w3 = g_ema.style(fix_z3)

						if args.starting_height_size==32:
							num_w = 9 if args.scene_size[-1]==512 else 7
						elif args.starting_height_size==4:
							num_w = 15 if args.scene_size[-1]==512 else 13

						w_plus3 = torch.tensor(w3.repeat(1,num_w,1), dtype=torch.float32, device=device)
					else:
						w_plus3 = w_plus1*w_inter + w_plus2*(1-w_inter)

					output = g_ema(in_pat.unsqueeze(0), styles=w_plus3, noise=noise_fix, input_type='w+')['image']

					# save feature maps
					H.append(output[:,0:1,:,:].repeat(1,3,1,1))
					D.append((0.5*output[:,1:4,:,:]+0.5)**(1/2.2))
					R.append(output[:,4:5,:,:].repeat(1,3,1,1))
					if isMetallic:
						M.append(output[:,5:6,:,:].repeat(1,3,1,1))

					ren_fea = output*0.5+0.5
					light, light_pos, size = set_param('cuda')
					fake_N = height_to_normal(ren_fea[:,0:1,:,:], intensity=args.H_scale)
					if isMetallic:
						ren_fea = torch.cat((2*fake_N-1,ren_fea[:,1:4,:,:],ren_fea[:,4:5,:,:].repeat(1,3,1,1), ren_fea[:,5:6,:,:].repeat(1,3,1,1)),dim=1)
					else:
						ren_fea = torch.cat((2*fake_N-1,ren_fea[:,1:4,:,:],ren_fea[:,4:5,:,:].repeat(1,3,1,1)),dim=1)
					tex_pos = getTexPos(ren_fea.shape[2], size, 'cuda').unsqueeze(0)
					rens = render(ren_fea, tex_pos, light, light_pos, isMetallic=isMetallic, no_decay=False) #[0,1]
					Ren.append(rens)

					# ren_fea = output*0.5+0.5
					# light, light_pos, size = set_param('cuda')
					# fake_N = height_to_normal(ren_fea[:,0:1,:,:], intensity=1)
					# ren_fea = torch.cat((2*torch.tile(fake_N,(1,1,2,2))-1,torch.tile(ren_fea[:,1:4,:,:],(1,1,2,2)), torch.tile(ren_fea[:,4:5,:,:].repeat(1,3,1,1),(1,1,2,2))),dim=1)
					# tex_pos = getTexPos(ren_fea.shape[2], size, 'cuda').unsqueeze(0)

					# print(ren_fea.shape)
					# print(tex_pos.shape)
					# rens_tile = render(ren_fea, tex_pos, light, light_pos, isSpecular=False, no_decay=False) #[0,1]
					# Ren_tile.append(rens_tile)

					w_inter += 1/N_inter


				H = torch.cat(H, dim=0)
				D = torch.cat(D, dim=0)
				R = torch.cat(R, dim=0)


				Ren = torch.cat(Ren, dim=0)

				# print(name, x.shape)
				save_path = os.path.join(out_path, save_idx+'_H.png')
				utils.save_image( H, save_path, nrow=N_inter+1, normalize=True, range=(-1,1) )

				save_path = os.path.join(out_path, save_idx+'_Ren.png')
				utils.save_image( Ren, save_path, nrow=N_inter+1, normalize=False)            

				save_path = os.path.join(out_path, save_idx+'_D.png')
				utils.save_image( D, save_path, nrow=N_inter+1, normalize=False)

				save_path = os.path.join(out_path, save_idx+'_R.png')
				utils.save_image( R, save_path, nrow=N_inter+1, normalize=True, range=(-1,1) )

				if isMetallic:
					M = torch.cat(M, dim=0)
					save_path = os.path.join(out_path, save_idx+'_M.png')
					utils.save_image( M, save_path, nrow=N_inter+1, normalize=True, range=(-1,1) )


def fixstyle(args, device, in_pats, idx):

	if args.video:
		WEAK=False
		STRONG=False
	else:
		WEAK=True
		STRONG=True

	N = 6
	save_dict = {0:'height_m',
				 1:'albedo_m',
				 2:'rough_m',
				 3:'height_std',
				 4:'albedo_std',
				 5:'rough_std',
	}

	# if args.dataset!='Tile':
	#     out_path = os.path.join('output', args.name, str(args.mode), args.savename)
	#     if not os.path.exists(out_path):
	#         os.makedirs(out_path)          
	#     fix_z = torch.randn( 1, args.style_dim, device=device)
	
	if args.add_jitter:
		jitter_origin = (torch.rand([1, 10], device=device)*2-1)*0.6
		jitter = jitter_origin.clone()
		input_std, input_m = torch.std_mean(in_pats, dim=(2,3))
		pats = torch.cat([input_m, input_std], dim=1)

	for n in range(N):

		out_path = os.path.join('output', args.name, str(args.mode), args.savename, str(n))
		if not os.path.exists(out_path):
			os.makedirs(out_path)  

		for index in range(in_pats.shape[0]):

			save_idx = str(index)
			in_pat=in_pats[index,...]
			# fix z for Tile dataset for each pattern
			fix_z = torch.randn( 1, args.style_dim, device=device)
			utils.save_image( in_pat.unsqueeze(0), os.path.join(out_path, save_idx+'inpat.png'), normalize=True, range=(-1,1))


			if args.color_cond:
				print(in_pat.shape)
				in_pat = color_pat(in_pat.squeeze(0).cpu().numpy())
				in_pat = torch.from_numpy(in_pat).permute(2,0,1).to(device).float()
				print(in_pat.dtype)

			g_ema = Generator(args,device).to(device)

			print("load ckpt: ", args.ckpt)
			ckpt = torch.load(os.path.join('output', args.name, 'checkpoint_eval', args.ckpt), map_location=lambda storage, loc: storage)

			g_ema.load_state_dict(ckpt["g_ema"])
			noise = g_ema.make_noise()

			## for weak 
			if WEAK:
				D = []
				H = []
				R = []
				Ren = []
				Pats = []

				for j in range(9):

					if args.add_jitter and args.dataset!='Tile':
						if n==0: # jitter height_m
							jitter[:,0:1] = torch.rand([1, 1], device=device)*2-1
						elif n==1: # jitter albedo_m
							jitter[:,1:4] = torch.rand([1, 3], device=device)*2-1
							# jitter[:,0:1] = torch.rand([1, 1], device=device)*0.6

						elif n==2: # jitter rough_m
							jitter[:,4:5] = torch.rand([1, 1], device=device)*2-1
						elif n==3: # jitter height_std
							jitter[:,5:6] = torch.rand([1, 1], device=device)*2-1
						elif n==4: # jitter albedo_std
							jitter[:,6:9] = torch.rand([1, 3], device=device)*2-1
						elif n==5: # jitter rough_std
							jitter[:,9:10] = torch.rand([1, 1], device=device)*2-1
						print(n, jitter)
					else:
						jitter = None

					# weak shift in_pats
					if args.dataset=='Tile':
						rand1 = [random.randint(0,512), random.randint(0,512)]
						in_pats_aug = mycrop(in_pat.unsqueeze(0), in_pat.shape[-1], rand0=rand1)
						shiftN = rand1 if args.shiftN else None
					else:
						in_pats_aug = in_pat.unsqueeze(0)
						shiftN = None

					if not args.nocond_z:
						output = g_ema(in_pats_aug, noise=noise, jitter=jitter)['image']  
					else:
						output = g_ema( in_pats_aug, styles=[fix_z], noise=noise, input_type='z' , shiftN=shiftN, jitter=jitter)['image']    

					# save feature maps
					H.append(output[:,0:1,:,:].repeat(1,3,1,1))
					D.append((0.5*output[:,1:4,:,:]+0.5)**(1/2.2))
					R.append(output[:,4:5,:,:].repeat(1,3,1,1))

					ren_fea = output*0.5+0.5
					light, light_pos, size = set_param('cuda')
					fake_N = height_to_normal(ren_fea[:,0:1,:,:])
					ren_fea = torch.cat((2*fake_N-1,ren_fea[:,1:4,:,:],ren_fea[:,4:5,:,:].repeat(1,3,1,1)),dim=1)
					tex_pos = getTexPos(ren_fea.shape[2], size, 'cuda').unsqueeze(0)
					rens = render(ren_fea, tex_pos, light, light_pos, isMetallic=False, no_decay=False) #[0,1]
					Ren.append(rens)


					if args.dataset!='Tile' and args.add_jitter:
						temp_pats = pats + jitter
						print('pats1 ',pats.shape)
						Pats.append(temp_pats.unsqueeze(-1).unsqueeze(-1))

				if args.dataset!='Tile' and args.add_jitter:

					jitter = jitter_origin.clone()

					Pats = torch.cat(Pats, dim=0)
					print('pats ',Pats.shape)
					temp_m = torch.cat([Pats[:,0:1,:,:].repeat(1,3,512,512), 2*((Pats[:,1:4,:,:].repeat(1,1,512,512)+1)*0.5)**(1/2.2)-1, Pats[:,4:5,:,:].repeat(1,3,512,512)], dim=-1)
					temp_std = torch.cat([Pats[:,5:6,:,:].repeat(1,3,512,512), 2*((Pats[:,6:9,:,:].repeat(1,1,512,512)+1)*0.5)**(1/2.2)-1, Pats[:,9:10,:,:].repeat(1,3,512,512)], dim=-1)
					utils.save_image( temp_m, os.path.join(out_path, save_idx+'_condm.png'), nrow=3,normalize=True, range=(-1,1))
					utils.save_image( temp_std, os.path.join(out_path, save_idx+'_condstd.png'), nrow=3,normalize=True, range=(-1,1))

				H = torch.cat(H, dim=0)
				D = torch.cat(D, dim=0)
				R = torch.cat(R, dim=0)

				Ren = torch.cat(Ren, dim=0)

				# print(name, x.shape)
				save_path = os.path.join(out_path, save_idx+'_H_weak.png')
				utils.save_image( H, save_path, nrow=3, normalize=True, range=(-1,1) )

				# save_tile_path = os.path.join(out_path, 'pat'+str(index)+'_Htile_weak.png')
				# utils.save_image( H_tile, save_tile_path, nrow=6, normalize=True, range=(-1,1) )

				save_path = os.path.join(out_path, save_idx+'_Ren_weak.png')
				utils.save_image( Ren, save_path, nrow=3, normalize=False)

				save_path = os.path.join(out_path, save_idx+'_D_weak.png')
				utils.save_image( D, save_path, nrow=3, normalize=False)
				# save_path = os.path.join(out_path, 'pat'+str(index)+'_Dtile_weak.png')
				# utils.save_image( D_tile, save_path, nrow=6, normalize=False)

				save_path = os.path.join(out_path, save_idx+'_R_weak.png')
				utils.save_image( R, save_path, nrow=3, normalize=True, range=(-1,1) )
				# save_path = os.path.join(out_path, 'pat'+str(index)+'_Rtile_weak.png')
				# utils.save_image( R_tile, save_path, nrow=6, normalize=True, range=(-1,1) )

			## for strong 
			if STRONG:

				D = []
				H = []
				R = []
				Ren = []
				Pats = []

				for j in range(9):

					in_pat=in_pats[random.randint(0, in_pats.shape[0]-1),...]

					# if args.dataset=='Tile':
					#     utils.save_image( in_pat.unsqueeze(0), os.path.join(out_path, save_idx+'inpat.png'), normalize=True, range=(-1,1))

					jitter = None

					# weak shift in_pats
					if args.dataset=='Tile':
						rand1 = [random.randint(0,512), random.randint(0,512)]
						in_pats_aug = mycrop(in_pat.unsqueeze(0), in_pat.shape[-1], rand0=rand1)
						shiftN = rand1 if args.shiftN else None
					else:
						in_pats_aug = in_pat.unsqueeze(0)
						shiftN = None

					if not args.nocond_z:
						output = g_ema(in_pats_aug, noise=noise, jitter=jitter)['image']  
					else:
						output = g_ema( in_pats_aug, styles=[fix_z], noise=noise, input_type='z' , shiftN=shiftN, jitter=jitter)['image']    

					# save feature maps
					H.append(output[:,0:1,:,:].repeat(1,3,1,1))
					D.append((0.5*output[:,1:4,:,:]+0.5)**(1/2.2))
					R.append(output[:,4:5,:,:].repeat(1,3,1,1))

					ren_fea = output*0.5+0.5
					light, light_pos, size = set_param('cuda')
					fake_N = height_to_normal(ren_fea[:,0:1,:,:])
					ren_fea = torch.cat((2*fake_N-1,ren_fea[:,1:4,:,:],ren_fea[:,4:5,:,:].repeat(1,3,1,1)),dim=1)
					tex_pos = getTexPos(ren_fea.shape[2], size, 'cuda').unsqueeze(0)
					rens = render(ren_fea, tex_pos, light, light_pos, isMetallic=False, no_decay=False) #[0,1]
					Ren.append(rens)


				H = torch.cat(H, dim=0)
				D = torch.cat(D, dim=0)
				R = torch.cat(R, dim=0)

				Ren = torch.cat(Ren, dim=0)

				# print(name, x.shape)
				save_path = os.path.join(out_path, save_idx+'_H_strong.png')
				utils.save_image( H, save_path, nrow=3, normalize=True, range=(-1,1) )

				# save_tile_path = os.path.join(out_path, 'pat'+str(index)+'_Htile_weak.png')
				# utils.save_image( H_tile, save_tile_path, nrow=6, normalize=True, range=(-1,1) )

				save_path = os.path.join(out_path, save_idx+'_Ren_strong.png')
				utils.save_image( Ren, save_path, nrow=3, normalize=False)

				save_path = os.path.join(out_path, save_idx+'_D_strong.png')
				utils.save_image( D, save_path, nrow=3, normalize=False)
				# save_path = os.path.join(out_path, 'pat'+str(index)+'_Dtile_weak.png')
				# utils.save_image( D_tile, save_path, nrow=6, normalize=False)

				save_path = os.path.join(out_path, save_idx+'_R_strong.png')
				utils.save_image( R, save_path, nrow=3, normalize=True, range=(-1,1) )
				# save_path = os.path.join(out_path, 'pat'+str(index)+'_Rtile_weak.png')
				# utils.save_image( R_tile, save_path, nrow=6, normalize=True, range=(-1,1) )


			## for weak 
			if args.video:

				out_temp_path = os.path.join(out_path, 'temp')
				if not os.path.exists(out_temp_path):
					os.makedirs(out_temp_path)

				gif=[]

				x_shift = 0
				y_shift = 0

				i=0
				while True:

					if x_shift>500 and y_shift>500:
						break

					jitter = None

					# weak shift in_pats
					rand1 = [x_shift, y_shift]
					print('x: ', x_shift, 'y: ', y_shift)
					in_pats_aug = mycrop(in_pat.unsqueeze(0), in_pat.shape[-1], rand0=rand1)
					shiftN = rand1 if args.shiftN else None

					if x_shift>500:
						y_shift+=10
					else:
						x_shift+=10

					if not args.nocond_z:
						output = g_ema(in_pats_aug, noise=noise, jitter=None)['image']  
					else:
						output = g_ema( in_pats_aug, styles=[fix_z], noise=noise, input_type='z' , shiftN=shiftN, jitter=None)['image']    

					ren_fea = output*0.5+0.5
					light, light_pos, size = set_param('cuda')
					fake_N = height_to_normal(ren_fea[:,0:1,:,:])
					ren_fea = torch.cat((2*fake_N-1,ren_fea[:,1:4,:,:],ren_fea[:,4:5,:,:].repeat(1,3,1,1)),dim=1)
					tex_pos = getTexPos(ren_fea.shape[2], size, 'cuda').unsqueeze(0)
					rens = render(ren_fea, tex_pos, light, light_pos, isMetallic=False, no_decay=False).squeeze(0).permute(1,2,0) #[0,1]
					# Ren.append(rens)

					imageio.imwrite(os.path.join(out_temp_path,  save_idx+'_ren_%02d.png'%i), (rens.cpu().numpy() * 255.0).astype(np.uint8))

					i+=1
				#     print(rens.shape)
				#     gif_numpy = rens.cpu().numpy()*255.0
				#     gif_numpy = np.clip(gif_numpy, 0, 255)
				#     gif_numpy = gif_numpy.astype(np.uint8)
				#     gif.append(gif_numpy)


				# save_gif_path = os.path.join(out_path, save_idx+'.gif')
				# imageio.mimwrite(save_gif_path, gif, format='GIF', fps=24)
				# print('save the gif',index)

				inputs = os.path.join(out_temp_path, save_idx+'_ren_%02d.png') 
				outputVideoPath = os.path.join(out_path, save_idx+'.mp4')   
				subprocess.run(["ffmpeg", "-r", "24", "-i", inputs, "-c:v", "libx264", "-vf", "fps=24", "-pix_fmt", "yuv420p", outputVideoPath])

				shutil.rmtree(out_temp_path)


def debug(args, device, in_pats):

	g_ema = Generator(args,device).to(device)

	print("load ckpt: ", args.ckpt)
	ckpt = torch.load(os.path.join('output', args.name, 'checkpoint_eval', args.ckpt), map_location=lambda storage, loc: storage)

	fix_z = torch.randn( 1, args.style_dim, device=device)

	g_ema.load_state_dict(ckpt["g_ema"])
	noise = g_ema.make_noise()

	in_pat=[]
	H = []
	H2 = []
	H3 = []
	R = []
	debug_sum=0
	for j in range(64):
		# print(j)

		# weak shift in_pats
		# rand1 = [random.randint(0,511), random.randint(0,511)]
		# reverse_rand1 = [512-rand1[0],512-rand1[1]]

		in_pats_aug = mycrop(in_pats.unsqueeze(0), in_pats.shape[-1])

		output = g_ema( in_pats_aug, styles=[fix_z], noise=noise, input_type='z' )['image']    
		# output2 = mycrop(output, in_pats.shape[-1], rand0=reverse_rand1)

		# if j>0:
		#     debug_sum += torch.sum(old_output2-output2)

		# old_output2 = output2

		# save feature maps
		in_pat.append(in_pats_aug[:,0:1,:,:].repeat(1,3,1,1))
		H.append(output[:,0:1,:,:].repeat(1,3,1,1))
		H2.append(output[:,8:9,:,:].repeat(1,3,1,1))
		H3.append(output[:,12:13,:,:].repeat(1,3,1,1))
		R.append(output[:,4:5,:,:].repeat(1,3,1,1))

	# print('debug_sum', debug_sum)
	in_pat = torch.cat(in_pat, dim=0)
	H = torch.cat(H, dim=0)
	H2 = torch.cat(H2, dim=0)
	H3 = torch.cat(H3, dim=0)
	R = torch.cat(R, dim=0)

	save_path = os.path.join(args.out_path, '_P_weak.png')
	utils.save_image( in_pat, save_path, nrow=8, normalize=True, range=(-1,1) )

	# print(name, x.shape)
	save_path = os.path.join(args.out_path, '_H_weak.png')
	utils.save_image( H, save_path, nrow=8, normalize=True, range=(-1,1) )

	save_path = os.path.join(args.out_path, '_H2_weak.png')
	utils.save_image( H2, save_path, nrow=8, normalize=True, range=(-1,1) )

	save_path = os.path.join(args.out_path, '_H3_weak.png')
	utils.save_image( H3, save_path, nrow=8, normalize=True, range=(-1,1) )

	save_path = os.path.join(args.out_path, '_R_weak.png')
	utils.save_image( R, save_path, nrow=8, normalize=True, range=(-1,1) )



if __name__ == "__main__":
	device = "cuda"

	# Do not specify any argument (except name) in CMD, I prefer to save all raw training files rather than args     
	parser = argparse.ArgumentParser()
	parser.add_argument("--name", type=str, help='name of the experiment. It decides where to store samples and models') 
	parser.add_argument("--savename", type=str, help='name of the experiment. It decides where to store samples and models') 
	parser.add_argument("--pat_path", type=str, default='', help='name of the experiment. It decides where to store samples and models') 
	parser.add_argument("--data_path", type=str, default='', help='name of the experiment. It decides where to store samples and models') 

	# Dataset related 
	parser.add_argument("--scene_size", type=int, default=512, help='size of data (H*W), used in defining dataset and model')
	parser.add_argument("--random_flip", type=bool, default=True, help='if random_flip or not')
	parser.add_argument("--random_crop", type=bool, default=False, help='if random_crop or not')
	parser.add_argument("--shuffle", type=bool, default=True, help='used in dataloader')
	# parser.add_argument('--isMetallic', action='store_true', help='is metal or not')        

	# Generator related 
	parser.add_argument("--style_dim", type=int, default=512)
	parser.add_argument("--n_mlp", type=int, default=8)     
	parser.add_argument("--channel_multiplier", type=int, default=2)
	parser.add_argument("--number_of_semantic", type=int, default=34, help='even including unlabled, i.e., all possible different int value could appear in raw sematic annotation')
	parser.add_argument("--have_zero_class", type=bool, default=True, help='Do you take 0 as one of semantic class label. If no we will shift semantic class by 1 as there will be no 0 class at all')
	parser.add_argument("--starting_height_size", type=int, default=32, help='encoder feature passed to generator, support 4,8,16,32.') 
	parser.add_argument("--condv", type=str, default='1', help='cond version: 1 || 2')

	# Loss weight related 
	parser.add_argument("--kl_lambda", type=float, default=0.01)
	parser.add_argument("--r1", type=float, default=10, help='loss weight for r1 regularization')
	parser.add_argument("--path_regularize", type=float, default=2, help='loss weight for path regularization')
	parser.add_argument("--vgg_regularize", type=float, default=1, help='loss weight for vgg regularization')
   
	# Training related
	parser.add_argument("--local_rank", type=int, default=0)
	parser.add_argument("--iter", type=int, default=800000, help='total number of iters for training')
	parser.add_argument("--ckpt_save_frenquency", type=int, default=10000, help='iter frenquency to save checkpoint')
	parser.add_argument("--start_keeping_iter", type=int, default=30000, help='after this, saved ckpt will not be removed. See CheckpointSave for details')
	parser.add_argument("--batch_size", type=int, default=4, help='batch size')
	parser.add_argument("--n_sample", type=int, default=16, help='for visualization')  
	parser.add_argument("--steps", type=int, default=2000, help='steps for optimization')  
	parser.add_argument("--lr", type=float, default=0.002, help='learning rate')
	parser.add_argument("--weight_L1", type=float, default=0.1, help='weight of downsampled L1')
	parser.add_argument("--start_iter", type=int, default=0, help='starting iter')    
	parser.add_argument("--ckpt", type=str, default=None, help='path to sceneGAN training ckpt.')
	parser.add_argument("--augment", type=bool, default=False, help='apply non-leaking augmentation in adv training')  # TODO make it action 
	parser.add_argument("--augment_p", type=str, default='0.5,0.5,0', help='cutout, color and translation')  
	parser.add_argument("--d_reg_every", type=int, default=16, help='perform r1 regularization for every how many steps')
	parser.add_argument("--g_reg_every", type=int, default=4, help='perform path regularization for every how many steps')
	parser.add_argument("--vgg_reg_every", type=int, default=4, help='perform vgg regularization for every how many steps, if 0 then no vgg loss')
	parser.add_argument("--dk_size", type=int, default=3, help='perform vgg regularization for every how many steps, if 0 then no vgg loss')
	parser.add_argument("--vgg_fix_noise", type=bool, default=True, help='noise will be fixed when perform vgg loss')
	parser.add_argument('--aug_data', action='store_true', help='data augmentation')        
	parser.add_argument('--extract_model', action='store_true', help='extract model of tileable patterns from target images')        
	parser.add_argument('--tile_crop', action='store_true', help='extract model of tileable patterns from target images')        
	parser.add_argument('--nocond_z', action='store_true', help='randonly sample z from normal distribution')        
	parser.add_argument('--circular', action='store_true', help='circular version 1')        
	parser.add_argument('--circular2', action='store_true', help='circular version 2')        
	parser.add_argument('--color_cond', action='store_true', help='color condition')        
	parser.add_argument('--load_color', action='store_true', help='load color map')        
	parser.add_argument('--resize', action='store_true', help='resize')        
	parser.add_argument('--pad_optim', action='store_true', help='optimize padding')        
	parser.add_argument('--rand_start', action='store_true', help='randonly sample z from normal distribution')        
	parser.add_argument('--highres', action='store_true', help='high resolution')        
	parser.add_argument('--inter_z', action='store_true', help='interploate z or not')        
	parser.add_argument('--cut_rough', type=float, default=0.3, help='threshold the roughness to ')        

	parser.add_argument('--add_jitter', action='store_true', help='add_jitter')        
	parser.add_argument('--shiftN', action='store_true', help='shift noise or not')        
	parser.add_argument('--truncate_z', type=float, default=1.0, help='truncate_z')        
	parser.add_argument('--scalar_cond', action='store_true', help='using scalar value as condition')        
	parser.add_argument('--no_cond', action='store_true', help='no conditional styleGAN2') 
	parser.add_argument('--rand_cond', action='store_true', help='randomize condition scalar') 
	parser.add_argument('--shift_optim', action='store_true', help='shift optim or not') 
	parser.add_argument('--shift_optim_L1', action='store_true', help='shift optim L1 only or not') 
	parser.add_argument('--inter_shift', action='store_true', help='inter leave shift or not') 
	parser.add_argument("--down_size", type=int, default=16, help='downsampled size')
	parser.add_argument('--use_all_pat', action='store_true', help='use_all_pat') 

	parser.add_argument("--dataset", type=str, default='Leather', help='Dataset is Tile or leather')  
	parser.add_argument("--mode", type=str, default='project', help='project or sample')  
	parser.add_argument("--optim", type=str, default='wn', help='project or sample')  
	parser.add_argument("--loss", type=str, default='TD', help='TD or MG')  
	parser.add_argument('--match_D', action='store_true', help='match the diffuse map')        
	parser.add_argument('--opt_gamma', action='store_true', help='optimize gamma of rough')  
	parser.add_argument('--opt_scale', action='store_true', help='optimize light and H_N factor')  
	parser.add_argument('--video', action='store_true', help='save to video or not for gif image')  
	parser.add_argument("--noise_reg", type=str, default='0', help='0,1,2 for noise reg') 
	parser.add_argument("--init", type=str, default='mean', help='mean initialization') 

	parser.add_argument('--H_scale', type=float, default=1.0, help='scale of height')        

	args = parser.parse_args()

	args.scene_size = (args.scene_size, args.scene_size)
	args.isMetallic=True if args.dataset=='Metal' else False

	# args = parser.parse_args()
	if args.load_color:
		pat_path = './Data/Bricks_color_pat' 
	elif args.resize:
		pat_path = './Data/Bricks_resize_pat' 
	else:
		pat_path = args.pat_path if args.pat_path!='' else './Data/Bricks_test_pat_demo'

	print(args.mode)

	total_pats = []
	total_pats_dict= {}

	for pat in os.listdir(pat_path):

		pat_name = pat.split('.')[0]
		path = os.path.join(pat_path, pat)
		pat_pil = Image.open(path)

		toTensor = transforms.ToTensor()
		pat_tensor = toTensor(pat_pil)
		_,h,w = pat_tensor.shape

		if pat_tensor.dtype == torch.int32:
			pat_tensor = pat_tensor.float() / 65535.0

		pats = pat_tensor[0:1,:,0:h]

		if pats.shape[-1]!=args.scene_size[-1]:
			pats = F.interpolate(pats.unsqueeze(0), size=(args.scene_size[0], args.scene_size[1]), mode='bilinear').squeeze(0)

		pats = 2*pats-1 #[0,1]-->[-1,1]

		total_pats.append(pats.unsqueeze(0))
		total_pats_dict[pat_name]=pats.unsqueeze(0).to(device)

	total_pats = torch.cat(total_pats, dim=0).to(device)

	if args.mode=='sample':
		with torch.no_grad():
			sample(args, device, in_pats=total_pats, idx=0)

	elif args.mode=='fixstyle':
		with torch.no_grad():
			fixstyle(args, device, in_pats=total_pats, idx=0)

	elif args.mode=='interpolate':
		with torch.no_grad():
			interpolate(args, device, in_pats=total_pats, idx=0)

	else:

		# target_path = './Data/Tile' if not args.resize else './Data/ResizeBricks'
		# target_path = './Data/Bricks' if not args.resize else './Data/ResizeBricks'
		target_path = args.data_path if args.data_path!='' else './Data/Bricks'

		pat_N = 1 if args.no_cond else 4

		for idx, target in enumerate(os.listdir(target_path)):
			print(target)

			target_name = target.split('.')[0]

			# reconstruct test pats and eval pats
			eval_pats=[]
			opt_pat=[]
			rand_list_opt = []
			for i in range(pat_N):
				rand_list_opt.append(random.randint(0,len(total_pats_dict)-1))

			rand_list_eval = []
			for i in range(pat_N):
				rand_list_eval.append(random.randint(0,len(total_pats_dict)-1))

			for temp_idx, key in enumerate(total_pats_dict):

				if args.use_all_pat:
					opt_pat.append(total_pats_dict[key])
					eval_pats.append(total_pats_dict[key])
				else:
					# # for bricks
					if args.dataset=='Tile':
						if target_name in key:
							print('target ', target_name , 'key ',key)
							opt_pat.append(total_pats_dict[key])
						else:
							eval_pats.append(total_pats_dict[key])
					# for other except bricks
					else:
						if temp_idx in rand_list_opt:
							opt_pat.append(total_pats_dict[key])
						if temp_idx in rand_list_eval:
							eval_pats.append(total_pats_dict[key])


			if not opt_pat:
				print(target, 'not match patterns, skip')
				continue 

			eval_pats = torch.cat(eval_pats, dim=0).to(device)
			opt_pat = torch.cat(opt_pat, dim=0).to(device)


			args.target = os.path.join(target_path, target)
			for index in range(opt_pat.shape[0]):

				if args.no_cond:
					if index>0:
						continue
					args.out_path = os.path.join('output', args.name, str(args.mode), args.savename, target_name)
					if not os.path.exists(args.out_path):
						os.makedirs(args.out_path)
				else:
					args.out_path = os.path.join('output', args.name, str(args.mode), args.savename, target_name +'/pat'+str(index))
					if not os.path.exists(args.out_path):
						os.makedirs(args.out_path)

				if args.highres:
					highres_projector(args, device, in_pats=opt_pat[index,...]) 
				elif args.mode=='embed':
					embed(args, device, in_pats=opt_pat[index,...])
				else:
					args.embed_path = os.path.join('./output' ,args.name, 'embed.pt')
					projector(args, device, in_pats=opt_pat[index,...])   

			del eval_pats
			del opt_pat
			del rand_list_opt
			del rand_list_eval
