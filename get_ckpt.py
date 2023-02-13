
import argparse

import torch
import os

def process(args):

	load_path = os.path.join('./output', args.name, 'checkpoint', args.file)
	save_path = os.path.join('./output', args.name, 'checkpoint_eval')
	ckpt = torch.load(load_path)

	if not os.path.exists(save_path):
		os.makedirs(save_path)
 
	torch.save(
		{
			'g_ema': ckpt['g_ema'],
		},
		os.path.join(
			save_path,
			f'{args.file}'),
	)


if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--file", type=str, help='name of the experiment. It decides where to store samples and models') 
	parser.add_argument("--name", type=str, help='name of the experiment. It decides where to store samples and models') 
   
	args = parser.parse_args()
	


	process(args)