from config.refiner_cfg import RefinerConfig
import torch 
import pandas as pd 
from dataset.utils.functions import *
from dataset.image_dataset.datasourcer import DataSourcer
from PIL import Image 
from copy import deepcopy
from dataset.image_dataset.process_funs import get_background_data
import random
import torchvision.transforms.functional as TF





class Dataset(torch.utils.data.Dataset):
    "This is a regular dataset class"
    def __init__(self, args, train):
        self.args = args 
        self.train = train 
            
        # prepare datasourcer  
        names = RefinerConfig.TRAIN_DATASETS if train else RefinerConfig.TEST_DATASETS
        paths_list = get_path(names, train=train, refiner=True)
        self.sourcer = DataSourcer(paths_list, refiner=True)    
        
        print('total data: ', len(self.sourcer.img_bank) )


    def flip(self, img, sem, ins, composition):
        return  TF.hflip(img), TF.hflip(sem), TF.hflip(ins), TF.hflip(composition)
        

    def __getitem__(self, idx):
        
        # read raw images 
        composition = Image.open( self.sourcer.composition_bank[idx] ).convert('RGB') # Note: this is generated image, res is not same as previous data
        img = Image.open( self.sourcer.img_bank[idx] ).convert('RGB')
        sem = Image.open( self.sourcer.sem_bank[idx] )
        ins = Image.open( self.sourcer.ins_bank[idx] )


        # resize into args.scene_size (Note: make sure args.scene_size is same as res of composition) 
        img = img.resize( (self.args.scene_size,self.args.scene_size), Image.NEAREST ) 
        sem = sem.resize( (self.args.scene_size,self.args.scene_size), Image.NEAREST ) 
        ins = ins.resize( (self.args.scene_size,self.args.scene_size), Image.NEAREST ) 

        if self.args.random_flip and random.random()>0.5:
            img, sem, ins, composition = self.flip(img, sem, ins, composition)

        out = get_background_data(img, sem, ins, composition)
        scene_img = out['scene_img'].squeeze(0)
        scene_sem = out['scene_sem'].squeeze(0)
        scene_ins = out['scene_ins'].squeeze(0)
        composition = out['composition'].squeeze(0)
 
        return scene_img, scene_sem, scene_ins, composition
 

    def __len__(self):
        return len(self.sourcer.img_bank)
