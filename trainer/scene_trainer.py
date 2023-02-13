import os
import sys
import shutil
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from tensorboardX import SummaryWriter
from copy import deepcopy

from models.scene_model import Generator, Discriminator
from dataset.image_dataset import get_scene_dataloader
from misc.DiffAugment import DiffAugment

from trainer.utils import sample_data, ImageSaver, accumulate, sample_n_data, to_device, CheckpointSaver, mycrop
from criteria.gan import g_nonsaturating_loss, d_logistic_loss, g_path_regularize, d_r1_loss
from criteria.l1_l2 import L1_loss, L2_loss
from criteria.vgg import VGGLoss, VGGLoss_nogt
from distributed import get_rank, synchronize, reduce_loss_dict, reduce_sum, get_world_size     
import time

from trainer.render import render, set_param, getTexPos, height_to_normal
import random

# from optim_utils import TDLoss2

# from torch.optim.lr_scheduler import StepLR

def get_edges(t):
	ByteTensor = torch.cuda.ByteTensor
	edge = ByteTensor(t.size()).zero_()
	edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1]).byte()
	edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1]).byte()
	edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).byte()
	edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).byte()
	return edge.float()


def process_data(args, data, device):  

	data = to_device(data, device)
	out = {}

	out['real_img'] = data['scene_img']
	out['global_pri'] = data['scene_sem']

	# shift by 1 if needed. 
	# In bg training we do not do zero padding on semantic 
	# if not args.have_zero_class:
	#     data['scene_sem'] = data['scene_sem']-1 

	# convert sem into channel representation  
	# batch, _, height, width = data['scene_sem'].shape
	# all_zeros = torch.zeros( batch, args.number_of_semantic, height, width ).to(device)
	# data['scene_sem'] = all_zeros.scatter_(1, data['scene_sem'], 1.0)

	# concat with edge map 
	# out['global_pri'] = torch.cat( [data['scene_sem'], get_edges(data['scene_ins'])], dim=1 )
	
	return out 



class Trainer():
	def __init__(self, args, device):

		self.args = args
		self.device = device


		self.prepare_model()
		self.prepare_optimizer()
		self.prepare_dataloader()
		self.get_vgg_loss = VGGLoss()
		# self.get_td_loss = TDLoss2(self.device, 2, low_level=True)
		self.get_vggnogt_loss = VGGLoss_nogt()
		self.loss_dict = {}
		self.MSELoss = torch.nn.MSELoss()

		if self.args.ckpt:
			self.load_ckpt()  
		if self.args.distributed:
			self.wrap_module()

		self.L2 = L2_loss()

		if self.args.augment:
			augment_p = [float(s.strip()) for s in self.args.augment_p.split(',')]
			self.policy = [ ['cutout',augment_p[0]], ['color',augment_p[1]], ['translation',augment_p[2]] ]

		if get_rank() == 0:
			self.prepare_exp_folder() 
			self.image_train_saver = ImageSaver(os.path.join('output',self.args.name,'sample_train'), int(self.args.n_sample**0.5) )
			self.image_test_saver = ImageSaver(os.path.join('output',self.args.name,'sample_test'), int(self.args.n_sample**0.5) )
			self.ckpt_saver = CheckpointSaver( args, os.path.join('output',self.args.name,'checkpoint')  ) 
			self.ckpt_saver_eval = CheckpointSaver( args, os.path.join('output',self.args.name,'checkpoint_eval')  ) 
			self.writer = SummaryWriter( os.path.join('output',self.args.name,'Log') )   
			# self.prepare_visualization_data()       

		synchronize()
			  
   

	def prepare_exp_folder(self):
		path = os.path.join( 'output', self.args.name  )
		if os.path.exists(path):
			shutil.rmtree(path)
		os.makedirs(path)
		os.makedirs(path+'/checkpoint' )
		os.makedirs(path+'/checkpoint_eval' )
		os.makedirs(path+'/sample_train' )
		os.makedirs(path+'/sample_test' )
		os.makedirs(path+'/Log' )    
		shutil.copy2(sys.argv[0], path)


	def wrap_module(self):
		self.generator = DDP( self.generator, device_ids=[self.args.local_rank], output_device=self.args.local_rank, broadcast_buffers=False )
		self.netD = DDP( self.netD, device_ids=[self.args.local_rank], output_device=self.args.local_rank, broadcast_buffers=False )

	def prepare_model(self):
		self.generator = Generator(self.args,self.device).to(self.device)
		self.netD = Discriminator(self.args).to(self.device)
		self.g_ema = Generator(self.args,self.device).to(self.device)
		self.g_ema.eval()
		accumulate(self.g_ema, self.generator, 0)

		# print(self.generator)

	def prepare_optimizer(self):
		g_reg_ratio = self.args.g_reg_every / (self.args.g_reg_every + 1)    
		d_reg_ratio = self.args.d_reg_every / (self.args.d_reg_every + 1)  

		# self.optimizerG = optim.Adam([{'params':self.generator.encoder.parameters(), 'lr':self.args.lr*g_reg_ratio*0.1},
		#                               {'params':self.generator.convs.parameters()}, 
		#                               {'params':self.generator.to_rgbs.parameters()} 
		#                             ], lr=self.args.lr*g_reg_ratio, betas=(0**g_reg_ratio, 0.99**g_reg_ratio) 
		#                             )

		self.optimizerG = optim.Adam( self.generator.parameters(), lr=self.args.lr*g_reg_ratio, betas=(0**g_reg_ratio, 0.99**g_reg_ratio) )
		self.optimizerD = optim.Adam( self.netD.parameters(), lr=self.args.lr*d_reg_ratio*self.args.d_lr, betas=(0**d_reg_ratio, 0.99**d_reg_ratio) )
	 
	def load_ckpt(self):
	   
		print("load ckpt: ", self.args.ckpt)
		ckpt = torch.load(self.args.ckpt, map_location=lambda storage, loc: storage)
		self.generator.load_state_dict(ckpt["generator"])
		self.netD.load_state_dict(ckpt["netD"])
		self.g_ema.load_state_dict(ckpt["g_ema"])

		self.optimizerG.load_state_dict(ckpt["optimizerG"])
		self.optimizerD.load_state_dict(ckpt["optimizerD"])

		self.args.start_iter = ckpt["iters"]
	   
	def prepare_dataloader(self):
		# print('prepare dataloader..................')
		self.train_loader = sample_data( self.args, get_scene_dataloader(self.args, train=True)   )
		self.test_loader = sample_data( self.args,  get_scene_dataloader( self.args, train=False)  )

	def prepare_visualization_data(self):
		data = sample_n_data(self.args.n_sample, self.train_loader, self.args.batch_size)
		# print('prepare data: ', data.shape)
		self.train_sample = process_data( self.args, data, self.device )  
		# self.image_train_saver( self.train_sample['real_img'], 'real.png' )   
		data = sample_n_data(self.args.n_sample, self.test_loader, self.args.batch_size)   
		self.test_sample = process_data( self.args, data, self.device )
		self.image_test_saver( self.test_sample['real_img'], 'real.png' ) 
		self.fix_z = torch.randn( 1, self.args.style_dim, device=self.device).repeat(self.args.n_sample,1)

	def write_loss(self,count):
		for key in self.loss_dict:
			self.writer.add_scalar(  key, self.loss_dict[key], count  )

	def print_loss(self,count):
		print( str(count)+' iter finshed' )
		for key in self.loss_dict:
			print(key, self.loss_dict[key])
		print('time has been spent in seconds since you lunched this script ', time.time()-self.tic)
		print(' ')

	def visualize(self, count):

		# in_D
		if self.args.debug:
			self.image_train_saver( self.fake_img.detach(), str(count).zfill(6)+'_D.png' )
			if self.args.cond_D:
				if self.args.scalar_cond:
					self.image_train_saver( self.cond_D[:, :5 ,:,:] , str(count).zfill(6)+'_D_m.png', gamma=False )
					self.image_train_saver( self.cond_D[:, -5:,:,:] , str(count).zfill(6)+'_D_std.png', gamma=False )
				else:
					self.image_train_saver( self.cond_D, str(count).zfill(6)+'_D_condpat.png', gamma=True if self.args.color_cond else False )
			self.image_train_saver( self.data['real_img'], str(count).zfill(6)+'_D_realimg.png' )
		# if self.args.color_cond:
		#     self.image_train_saver( 2*((self.color_condD+1)*0.5)**(1/2.2)-1 , str(count).zfill(6)+'_D_color.png' )

		self.prepare_visualization_data()       
		with torch.no_grad():  
			### in training sets
			output = self.g_ema( self.train_sample['global_pri'])
			if not self.args.scalar_cond and not self.args.no_cond:
				self.image_train_saver( self.train_sample['global_pri'] , str(count).zfill(6)+'_eval_pat.png', gamma=True if self.args.color_cond else False )
			elif self.args.scalar_cond:
				self.image_train_saver( self.train_sample['global_pri'] , str(count).zfill(6)+'_eval_pat.png', gamma=True if self.args.color_cond else False )
				# comput mean and std
				input_std, input_m = torch.std_mean(self.train_sample['global_pri'], dim=(2,3), keepdim=True)
				self.image_train_saver( input_m.repeat(1,1,512,512) , str(count).zfill(6)+'_input_m.png', gamma=False )
				self.image_train_saver( input_std.repeat(1,1,512,512) , str(count).zfill(6)+'_input_std.png', gamma=False )

			self.image_train_saver( self.train_sample['real_img'] , str(count).zfill(6)+'_eval_real.png' )
			# if self.args.debug:
			self.image_train_saver( output['image'] , str(count).zfill(6)+'_eval.png' )
			# self.image_train_saver( torch.tile(output['image'], (1,1,2,2)), str(count).zfill(6)+'_eval_tile.png' ) # save tiled image

			# save rendered image
			real = self.train_sample['real_img']
			fake = output['image']

			fake_fea_vgg = fake*0.5+0.5
			real_fea_vgg = real*0.5+0.5

			light, light_pos, size = set_param('cuda')

			fake_N = height_to_normal(fake_fea_vgg[:,0:1,:,:])
			fake_fea = torch.cat((2*fake_N-1,fake_fea_vgg[:,1:4,:,:],fake_fea_vgg[:,4:5,:,:].repeat(1,3,1,1)),dim=1)
			tex_pos = getTexPos(fake_fea.shape[2], size, 'cuda').unsqueeze(0)
			fake_rens = render(fake_fea, tex_pos, light, light_pos, isMetallic=False, no_decay=False) #[0,1]

			real_N = height_to_normal(real_fea_vgg[:,0:1,:,:])
			real_fea = torch.cat((2*real_N-1,real_fea_vgg[:,1:4,:,:],real_fea_vgg[:,4:5,:,:].repeat(1,3,1,1)),dim=1)
			# tex_pos = getTexPos(ren_fea.shape[2], size, 'cuda').unsqueeze(0)
			real_rens = render(real_fea, tex_pos, light, light_pos, isMetallic=False, no_decay=False) #[0,1]

			if get_rank()==0 and count%self.args.save_img_freq==0:
				self.image_train_saver( 2*fake_rens-1, f'{str(count).zfill(6)}_eval_fake_rens.png' )   
				self.image_train_saver( 2*real_rens-1, f'{str(count).zfill(6)}_eval_real_rens.png' )   

			### in testing sets
			output = self.g_ema( self.test_sample['global_pri']) 
			# if not self.args.scalar_cond and not self.args.no_cond:
			# self.image_test_saver( self.test_sample['global_pri'] , str(count).zfill(6)+'_eval_pat.png', gamma=True if self.args.color_cond else False  )
			self.image_test_saver( self.test_sample['real_img'] , str(count).zfill(6)+'_eval_real.png' )
			self.image_test_saver( output['image'] , str(count).zfill(6)+'_eval.png' )
			# self.image_test_saver( torch.tile(output['image'], (1,1,2,2)), str(count).zfill(6)+'_eval_tile.png' ) # save tiled image

			# fix style keep changing patterns
			if self.args.nocond_z and not self.args.no_cond:
				output = self.g_ema( self.train_sample['global_pri'], styles=[self.fix_z], input_type='z' )    
				self.image_train_saver( output['image'] , str(count).zfill(6)+'_eval_fix.png' )
				# self.image_train_saver( torch.tile(output['image'], (1,1,2,2)), str(count).zfill(6)+'_eval_fix_tile.png' ) # save tiled image

			# if self.args.color_cond:
			#     self.image_train_saver( 2*((self.train_sample['color_pri']+1)*0.5)**(1/2.2)-1 , str(count).zfill(6)+'_eval_color.png' )
			#     self.image_test_saver( 2*((self.test_sample['color_pri']+1)*0.5)**(1/2.2)-1 , str(count).zfill(6)+'_eval_color.png' )

	def visualize_train(self, count, output):

		# tr
		if not self.args.scalar_cond and not self.args.no_cond:
			self.image_train_saver( self.data['global_pri'] , str(count).zfill(6)+'_tr_pat.png', gamma=True if self.args.color_cond else False  )
		self.image_train_saver( self.data['real_img'] , str(count).zfill(6)+'_tr_img.png' )
		# if self.args.color_cond:
		#     self.image_train_saver( self.data['color_pri'] , str(count).zfill(6)+'_tr_color.png' )
		self.image_train_saver( output['image'] , str(count).zfill(6)+'_tr.png' )
		self.image_train_saver( torch.tile(output['image'], (1,1,2,2)), str(count).zfill(6)+'_tr_tile.png' ) # save tiled image

	def save_ckpt(self, count):

		save_dict =  {   "args": self.args,
						 "generator": self.g_module.state_dict(),
						 "netD": self.d_module.state_dict(),
						 "g_ema": self.g_ema.state_dict(),
						 "optimizerG": self.optimizerG.state_dict(),
						 "optimizerD": self.optimizerD.state_dict(),
						 "iters": count }
		self.ckpt_saver( save_dict, count )

		save_dict =  {"g_ema": self.g_ema.state_dict()}
		self.ckpt_saver_eval( save_dict, count )

	def trainD(self):

		if self.args.augment:
			augmented_real_img = DiffAugment( deepcopy(self.data['real_img']), self.policy )
			augmented_fake_img = DiffAugment( deepcopy(self.fake_img.detach()), self.policy )
		else:
			augmented_real_img = self.data['real_img']
			augmented_fake_img = self.fake_img.detach()

		if self.args.cond_D:
			augmented_fake_img = torch.cat([augmented_fake_img, self.cond_D], dim=1)
			augmented_real_img = torch.cat([augmented_real_img, self.cond_D], dim=1)

		# if self.args.color_cond:
		#     augmented_fake_img = torch.cat([augmented_fake_img, self.color_condD], dim=1)
		#     augmented_real_img = torch.cat([augmented_real_img, self.color_condD], dim=1)

		real_pred = self.netD(augmented_real_img)
		fake_pred = self.netD(augmented_fake_img) 
		# print('trainD real_pred ',real_pred.isnan().any())
		# print('trainD fake_pred ',fake_pred.isnan().any())

		d_loss = d_logistic_loss(real_pred, fake_pred)  
		self.loss_dict["d"] = d_loss.item()

		self.optimizerD.zero_grad()
		d_loss.backward()
		self.optimizerD.step()

		for name, params in self.netD.named_parameters():
			assert params.isnan().any()==False, f'nan in {name} trainD'

	def Compute_scalar_reg(self, cond_D, out):
		gt_std, gt_m = torch.std_mean(cond_D, dim=(2,3))
		out_std, out_m = torch.std_mean(out, dim=(2,3))

		return (self.L2(gt_std,out_std) + self.L2(gt_m,out_m))*0.5


	def trainG(self):
	   
		if self.args.augment:
			augmented_fake_img = DiffAugment( self.fake_img, self.policy )
		else:
			augmented_fake_img = self.fake_img

		if self.args.cond_D:
			augmented_fake_img = torch.cat([augmented_fake_img, self.cond_D], dim=1)
		# if self.args.color_cond:
		#     augmented_fake_img = torch.cat([augmented_fake_img, self.color_condD], dim=1)

		if self.args.scalar_cond and self.args.scalar_regularize!=0:
			stat_loss = self.Compute_scalar_reg(self.cond_D, self.fake_img) * self.args.scalar_regularize
			self.loss_dict["scalar_reg"] = stat_loss.item()
		else:
			stat_loss = 0

		for name, params in self.netD.named_parameters():
			assert params.isnan().any()==False, f'nan in {name} trainG'

		# print('train G')
		fake_pred = self.netD(augmented_fake_img)
		# print('trainG fake_pred ',fake_pred.isnan().any())

		g_loss = g_nonsaturating_loss(fake_pred)
		self.loss_dict["g"] = g_loss.item()

		loss = g_loss + self.kl_loss + stat_loss


		# for name, params in self.generator.named_parameters():
		#     if 'encoder' in name:
		#         print(name,params.mean())

		self.optimizerG.zero_grad()
		loss.backward()
		self.optimizerG.step()

		# for name, params in self.generator.named_parameters():
		#     if 'encoder' in name:
		#         print('grad ', name,params.grad.mean())
		#         print('update after grad',name,params.mean())

	def regularizeD(self):


		if self.args.cond_D:       
		    in_D = torch.cat([self.data['real_img'], self.cond_D], dim=1)   
		else:
		    in_D = self.data['real_img']
		in_D.requires_grad = True

		# if self.args.color_cond:
		#     in_D = torch.cat([in_D, self.color_condD], dim=1)

		real_pred = self.netD(in_D)
		# print('regularizeD real_pred ',real_pred.isnan().any())

		# r1_loss = d_r1_loss(real_pred, self.data['real_img'])
		r1_loss = d_r1_loss(real_pred, in_D)
		# print('regularizeD r1_loss ',r1_loss.isnan().any())
		r1_loss = self.args.r1 / 2 * r1_loss * self.args.d_reg_every + 0 * real_pred[0]
		self.loss_dict["r1"] = r1_loss.item()
		assert r1_loss.isnan().any()==False, 'nan in r1_loss'
		assert r1_loss.isinf().any()==False, 'inf in r1_loss'

		self.optimizerD.zero_grad()
		r1_loss.backward()
		self.optimizerD.step()

		# print(r1_loss, r1_loss.isnan().any(), r1_loss.isinf().any())

		# for name, params in self.netD.named_parameters():
		#     assert params.grad.isinf().any()==False, f'inf in {name} grad regularizeD'
		#     assert params.grad.isnan().any()==False, f'nan in {name} grad regularizeD'
		#     assert params.isnan().any()==False, f'nan in {name} regularizeD'

	def regularizePath(self):


		output = self.generator(self.data['global_pri'], return_latents=True, return_loss=False)
		fake_img = output['image']
		latents = output['latent']

		path_loss, self.mean_path_length = g_path_regularize( fake_img, latents, self.mean_path_length )
		path_loss = self.args.path_regularize * self.args.g_reg_every * path_loss + 0 * fake_img[0, 0, 0, 0]
		self.loss_dict['path_loss'] = path_loss.item()

		# for name, params in self.generator.named_parameters():
		#     if 'encoder' in name:
		#         print('before update',name,params.mean())
		assert path_loss.isnan().any()==False, f'nan in {name} path_loss'

		self.optimizerG.zero_grad()
		path_loss.backward()
		self.optimizerG.step()

		# for name, params in self.generator.named_parameters():
		#     if 'encoder' in name:
		#         print('grad ', name,params.grad.mean())
		#         print('after update',name,params.mean())

	def preVGG(self, x):
		mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
		std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
		return (x-mean)/std

	def regularizeVGG(self, count, rand0=None, nocrop_real=None):
		randomize_noise = not self.args.vgg_fix_noise
		output = self.generator(self.data['global_pri'], return_loss=False, randomize_noise=randomize_noise)
		fake_img = output['image']

		if self.args.tile_crop:
			fake_crop = mycrop(fake_img, fake_img.shape[-1], rand0=rand0)
		else:
			fake_crop = fake_img
		real_crop = self.data['real_img']

		# rerender if necessary
		if fake_img.shape[1] == 5 or fake_img.shape[1] == 6:
			fake_fea_vgg = fake_crop*0.5+0.5
			real_fea_vgg = real_crop*0.5+0.5

			light, light_pos, size = set_param('cuda')

			fake_N = height_to_normal(fake_fea_vgg[:,0:1,:,:])
			fake_fea = torch.cat((2*fake_N-1,fake_fea_vgg[:,1:4,:,:],fake_fea_vgg[:,4:5,:,:].repeat(1,3,1,1)),dim=1)
			tex_pos = getTexPos(fake_fea.shape[2], size, 'cuda').unsqueeze(0)
			fake_rens = render(fake_fea, tex_pos, light, light_pos, isMetallic=False, no_decay=False) #[0,1]

			real_N = height_to_normal(real_fea_vgg[:,0:1,:,:])
			real_fea = torch.cat((2*real_N-1,real_fea_vgg[:,1:4,:,:],real_fea_vgg[:,4:5,:,:].repeat(1,3,1,1)),dim=1)
			# tex_pos = getTexPos(ren_fea.shape[2], size, 'cuda').unsqueeze(0)
			real_rens = render(real_fea, tex_pos, light, light_pos, isMetallic=False, no_decay=False) #[0,1]

			if get_rank()==0 and count%self.args.save_img_freq==0:
				self.image_train_saver( 2*fake_rens-1, f'{str(count).zfill(6)}_vg_fake_rens.png' )   
			#     self.image_train_saver( 2*real_rens-1, f'{str(count).zfill(6)}_vg_real_rens.png' )   

			vgg_loss = self.get_vgg_loss(self.preVGG(fake_rens), self.preVGG(real_rens)) * self.args.vgg_reg_every * self.args.vgg_regularize

		elif fake_img.shape[1] == 3:
			vgg_loss = self.get_vgg_loss(fake_img, self.data['real_img']) * self.args.vgg_reg_every * self.args.vgg_regularize

		self.loss_dict['vgg_loss'] = vgg_loss.item()

		self.optimizerG.zero_grad()
		vgg_loss.backward()
		self.optimizerG.step()

	def regularize_style(self, count):

		# random produce one style
		# random_noise = self.args.random_noise_style
		if True:
			fix_z = torch.randn( self.args.batch_size, self.args.style_dim, device=self.device)
			# if get_rank()==0:
			fix_noise = self.generator.module.make_noise()
			# fix_noise = self.generator.make_noise()

			w = self.data['global_pri'].shape[-1]
			output1 = self.generator(self.data['global_pri'], styles=[fix_z], noise=fix_noise, input_type='z' ,return_loss=False )['image']

			rand1 = [random.randint(1,w-1), random.randint(1,w-1)]

			shiftN=rand1 if self.args.shiftN else None
			# print('rand1: ', rand1)
			reverse_rand1 = [w-rand1[0],w-rand1[1]]
			# print('reverse_rand1: ', reverse_rand1)

			crop_pat = mycrop(self.data['global_pri'], w, rand0=rand1)

			output2 = self.generator(crop_pat, styles=[fix_z], noise=fix_noise, input_type='z' ,return_loss=False, shiftN=shiftN)['image']

			re_output2 = mycrop(output2, w, rand0=reverse_rand1)
			crop_pat2 = mycrop(crop_pat, w, rand0=reverse_rand1)

			style_loss = self.MSELoss(output1, re_output2) * self.args.style_reg_every * self.args.style_regularize

			self.loss_dict['style_loss'] = style_loss.item()

			self.optimizerG.zero_grad()
			style_loss.backward()
			self.optimizerG.step()

		else:
			fix_z = torch.randn( 1, self.args.style_dim, device=self.device)
			# if get_rank()==0:
			# fix_noise = self.generator.module.make_noise()
			# fix_noise = self.generator.make_noise()

			print('global pri: ',self.data['global_pri'].shape)
			print('fix_z: ',fix_z.shape)
			# print('fix_noise: ',fix_noise.shape)
			w = self.data['global_pri'].shape[-1]
			output1 = self.generator(self.data['global_pri'][0:1,...], styles=[fix_z], input_type='z' ,return_loss=False)['image']

			output2 = self.generator(self.data['global_pri'][1:2,...], styles=[fix_z], input_type='z' ,return_loss=False)['image']
  
			# print('output1: ',output1.shape)
			# print('output2: ',output2.shape)

			# criterion_D = TDLoss(output1[:,1:4,...], self.device, 2, low_level=True)
			# criterion_R = TDLoss(output1[:,4:5,...], self.device, 2, low_level=True)


			if get_rank()==0 and count%self.args.save_img_freq==0:

				self.image_train_saver( output1, str(count).zfill(6)+'_fea1.png' )
				self.image_train_saver( output2, str(count).zfill(6)+'_fea2.png' )

				# self.image_train_saver( self.data['global_pri'][0:2,...], f'{str(count).zfill(6)}_pat0.png', gamma=True if self.args.color_cond else False  )   

			style_loss = (self.get_td_loss(output2[:,1:4,...],output1[:,1:4,...] ) + self.get_td_loss(output2[:,4:5,...], output1[:,4:5,...])) * self.args.style_reg_every * self.args.style_regularize
			# style_loss = self.MSELoss(output2,output1 ) * self.args.style_reg_every * self.args.style_regularize

			self.loss_dict['style_loss'] = style_loss.item()

			self.optimizerG.zero_grad()
			style_loss.backward()
			self.optimizerG.step()			

			# del criterion_D
			# del criterion_R
			# del output1
			# del output2

	def regularize_color(self, count):

		output = self.generator(self.data['global_pri'] ,return_loss=False )['image']
		output1 = self.generator(self.data['global_pri'] ,return_loss=False )['image']

		if get_rank()==0 and count%self.args.save_img_freq==0:

			self.image_train_saver( output1 , str(count).zfill(6)+'_colorfea1.png' )
			self.image_train_saver( output, str(count).zfill(6)+'_colorfea2.png' )
			if not self.args.scalar_cond:
				self.image_train_saver( self.data['global_pri'], f'{str(count).zfill(6)}_colorpat0.png', gamma=True if self.args.color_cond else False  )   
  
		color_loss = -self.MSELoss(output1, output) * self.args.color_reg_every * self.args.color_regularize

		self.loss_dict['color_loss'] = color_loss.item()

		self.optimizerG.zero_grad()
		color_loss.backward()
		self.optimizerG.step()
   
		
   
	def train(self):
		"Note that in dist training printed and saved losses are not reduced, but from the first process"

		self.mean_path_length = 0
		self.tic = time.time()

		if self.args.distributed:
			self.g_module = self.generator.module
			self.d_module = self.netD.module
		else:
			self.g_module = self.generator
			self.d_module = self.netD


		for idx in range(self.args.iter):
			count = idx + self.args.start_iter

			if get_rank()==0:
				print('step ', count, '...learning rate.....', self.optimizerG.param_groups[0]['lr'], 'dk_size', self.args.dk_size)

			if count > self.args.iter:
				print("Done!")
				break

			self.data = process_data( self.args, next(self.train_loader), self.device )

			# for name, params in self.generator.named_parameters():
			#     if 'encoder' in name:
			#         print(name,params.mean())

			# forward G  
			output = self.generator(self.data['global_pri'])

			self.fake_img = output['image']
			self.kl_loss = output['klloss']*self.args.kl_lambda
			self.loss_dict['kl'] = self.kl_loss.item()


			assert self.fake_img.isnan().any()==False, 'nan self.fake_img'

			if get_rank()==0 and count % self.args.save_img_freq == 0:
				self.visualize_train(count, output)

			# crop to Discriminator
			if self.args.tile_crop:
				rand0 = [random.randint(-self.args.scene_size[0]+1, self.fake_img.shape[-1]-1),random.randint(-self.args.scene_size[0]+1, self.fake_img.shape[-1]-1)]
				self.fake_img = mycrop(self.fake_img, self.fake_img.shape[-1], rand0=rand0)
				self.data['real_img'] = mycrop(self.data['real_img'], self.fake_img.shape[-1], rand0=rand0)

				if self.args.cond_D or self.args.scalar_regularize!=0:
					self.cond_D = mycrop(self.data['global_pri'], self.fake_img.shape[-1], rand0=rand0)

				if self.args.scalar_cond:
					# compute the statistics of input
					input_std, input_m = torch.std_mean(self.cond_D, dim=(2,3))
					self.cond_D = torch.cat([input_m, input_std], dim=1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.fake_img.shape[-2], self.fake_img.shape[-1])
					# input = torch.cat([input1, input_cat], dim=1)
					# print(input.shape)

			else:
				rand0=None

			# update D
			self.trainD()
			if count % self.args.d_reg_every == 0 and self.args.r1!=0:
				self.regularizeD()

			# update G
			self.trainG()
			if count % self.args.g_reg_every == 0:
				self.regularizePath()
			if self.args.vgg_reg_every != 0 and count % self.args.vgg_reg_every == 0 and self.args.vgg_regularize!=0:
				self.regularizeVGG(count, rand0)
			if self.args.style_reg_every != 0 and count % self.args.style_reg_every == 0 and self.args.style_regularize!=0:
				self.regularize_style(count)
			if self.args.color_reg_every != 0 and count % self.args.color_reg_every == 0 and self.args.color_regularize!=0 and self.args.color_cond:
				self.regularize_color(count)


			accum = 0.5 ** (32 / (10 * 1000))
			accumulate(self.g_ema, self.g_module, accum)

			if get_rank() == 0:
				if count % 10 == 0:
					self.write_loss(count)
				if count % 50 == 0:
					self.print_loss(count)
				if count % self.args.save_img_freq == 0:
					self.visualize(count)
					# self.visualize_train(count, output)
				if count % self.args.ckpt_save_frenquency == 0:  
					self.save_ckpt(count)


			if count % self.args.lr_gamma_every ==0 and self.optimizerG.param_groups[0]['lr'] < self.args.lr_limit*self.args.g_reg_every / (self.args.g_reg_every + 1) and count!=0:
				self.optimizerG.param_groups[0]['lr'] += self.args.lr_gamma
				self.optimizerD.param_groups[0]['lr'] += self.args.lr_gamma

			synchronize()