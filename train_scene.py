import argparse
from trainer.scene_trainer import Trainer
import torch 
from distributed import synchronize
import os

def main(args, device):
    trainer = Trainer(args, device)
    trainer.train()



if __name__ == "__main__":
    device = "cuda"


    # Do not specify any argument (except name) in CMD, I prefer to save all raw training files rather than args     
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help='name of the experiment. It decides where to store samples and models') 

    # Dataset related 
    parser.add_argument("--scene_size", type=int, default=512, help='size of data (H*W), used in defining dataset and model')
    parser.add_argument("--random_flip", type=bool, default=True, help='if random_flip or not')
    parser.add_argument("--random_crop", type=bool, default=False, help='if random_crop or not')
    parser.add_argument("--shuffle", type=bool, default=True, help='used in dataloader')
    parser.add_argument("--dataset", type=str, default='Tile', help='which dataset we use')

    # Generator related 
    parser.add_argument("--style_dim", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)     
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--number_of_semantic", type=int, default=34, help='even including unlabled, i.e., all possible different int value could appear in raw sematic annotation')
    parser.add_argument("--have_zero_class", type=bool, default=True, help='Do you take 0 as one of semantic class label. If no we will shift semantic class by 1 as there will be no 0 class at all')
    parser.add_argument("--starting_height_size", type=int, default=32, help='encoder feature passed to generator, support 4,8,16,32.') 
    parser.add_argument("--rand_start", action='store_true', help='randomize starting layer or not')

    # Loss weight related 
    parser.add_argument("--kl_lambda", type=float, default=0.01)
    parser.add_argument("--r1", type=float, default=10, help='loss weight for r1 regularization')
    parser.add_argument("--path_regularize", type=float, default=2, help='loss weight for path regularization')
    parser.add_argument("--vgg_regularize", type=float, default=1, help='loss weight for vgg regularization')
    parser.add_argument("--style_regularize", type=float, default=1, help='loss weight for style regularization')
    parser.add_argument("--color_regularize", type=float, default=0, help='loss weight for color regularization')
    parser.add_argument("--scalar_regularize", type=float, default=0, help='loss weight for scalar regularization')
    parser.add_argument("--condv", type=str, default='1', help='cond version: 1 || 2')

    # Training related
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--iter", type=int, default=2000000, help='total number of iters for training')
    parser.add_argument("--ckpt_save_frenquency", type=int, default=10000, help='iter frenquency to save checkpoint')
    parser.add_argument("--save_img_freq", type=int, default=2000, help='iter frenquency to save imgs')
    parser.add_argument("--start_keeping_iter", type=int, default=30000, help='after this, saved ckpt will not be removed. See CheckpointSave for details')
    parser.add_argument("--batch_size", type=int, default=4, help='batch size')
    parser.add_argument("--n_sample", type=int, default=8, help='for visualization')  
    parser.add_argument("--lr", type=float, default=0.002, help='learning rate')
    parser.add_argument("--d_lr", type=float, default=1.0, help='learning rate scaling for discriminator')
    parser.add_argument("--start_iter", type=int, default=0, help='starting iter')    
    parser.add_argument("--ckpt", type=str, default=None, help='path to sceneGAN training ckpt.')
    parser.add_argument("--augment", type=bool, default=False, help='apply non-leaking augmentation in adv training')  # TODO make it action 
    parser.add_argument("--augment_p", type=str, default='0.5,0.5,0', help='cutout, color and translation')  
    parser.add_argument("--d_reg_every", type=int, default=16, help='perform r1 regularization for every how many steps')
    parser.add_argument("--g_reg_every", type=int, default=4, help='perform path regularization for every how many steps')
    parser.add_argument("--vgg_reg_every", type=int, default=4, help='perform vgg regularization for every how many steps, if 0 then no vgg loss')
    parser.add_argument("--style_reg_every", type=int, default=4, help='perform style regularization for every how many steps, if 0 then no vgg loss')
    parser.add_argument("--color_reg_every", type=int, default=4, help='perform color regularization for every how many steps, if 0 then no vgg loss')
    parser.add_argument("--vgg_fix_noise", type=bool, default=True, help='noise will be fixed when perform vgg loss')
    parser.add_argument('--aug_data', action='store_true', help='data augmentation')        
    parser.add_argument('--extract_model', action='store_true', help='extract model of tileable patterns from target images')        
    parser.add_argument('--tile_crop', action='store_true', help='extract model of tileable patterns from target images')        
    parser.add_argument('--nocond_z', action='store_true', help='randonly sample z from normal distribution')        
    parser.add_argument('--cond_D', action='store_true', help='conditional to z')        
    parser.add_argument('--circular', action='store_true', help='circular padding for all the conv, upsampling and downsampling operations')        
    parser.add_argument('--circular2', action='store_true', help='2nd version circular padding for all the conv, upsampling and downsampling operations')        
    parser.add_argument('--color_cond', action='store_true', help='add color condition')        
    parser.add_argument('--debug', action='store_true', help='add color condition')        
    parser.add_argument('--lr_gamma', type=float, default=5e-5, help='add color condition')        
    parser.add_argument('--truncate_z', type=float, default=1.0, help='truncate_z')        
    parser.add_argument('--lr_gamma_every', type=int, default=10000, help='add color condition')        
    parser.add_argument('--lr_limit', type=float, default=0.0008, help='lr limit')        

    parser.add_argument('--shiftN', action='store_true', help='use shift noise or not')        
    parser.add_argument('--scalar_cond', action='store_true', help='using scalar value as condition')        
    parser.add_argument('--no_cond', action='store_true', help='no conditional styleGAN2')        
    parser.add_argument('--rand_cond', action='store_true', help='randomize condition scalar') 
    parser.add_argument('--add_jitter', action='store_true', help='add_jitter')        

    parser.add_argument('--dk_size', type=int, default=3, help='downsample kernel size')        

    args = parser.parse_args()
    
    assert args.augment == False, 'augmentation is never tested. Not sure if it works...'

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    args.scene_size = (args.scene_size, args.scene_size)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
        args.sampler_seed = 1 # used in DistributedSampler, see https://github.com/pytorch/pytorch/blob/master/torch/utils/data/distributed.py

    main(args, device)
    
    # CUDA_VISIBLE_DEVICES=0,1,3,4 python -m torch.distributed.launch --nproc_per_node=4 train_scene.py city_branch2
