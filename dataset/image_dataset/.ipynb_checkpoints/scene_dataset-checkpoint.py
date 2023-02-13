from config.scene_cfg import SceneConfig
import torch 
import pandas as pd 
from dataset.utils.functions import *
from dataset.image_dataset.datasourcer import DataSourcer
from PIL import Image 
from copy import deepcopy
from dataset.image_dataset.process_funs import get_scene_data
import random
import torchvision.transforms.functional as TF





class Dataset(torch.utils.data.Dataset):
    "This is a regular dataset class"
    def __init__(self, args, train, datasets=None):
        "datasets is given when do composition otherwise you also need to specify SceneConfig when do compositon"
        self.args = args 
        self.train = train 
        
        print('NEAREST IS USED')
        #print('BICUBIC IS USED')
        import time
        time.sleep(2)
            
        # prepare datasourcer  
        if datasets is None:
            datasets = SceneConfig.TRAIN_DATASETS if train else SceneConfig.TEST_DATASETS
        paths_list = []
        for dataset in datasets:
            paths_list.append( get_path( dataset, train=train ) )
        self.sourcer = DataSourcer(paths_list)    
        

    def flip(self, img, sem, ins):
        return  TF.hflip(img), TF.hflip(sem), TF.hflip(ins)


    def resize(self, img, sem, ins, size):
        "size should be a tuple: W*H"
        img = img.resize( size, Image.NEAREST ) 
        sem = sem.resize( size, Image.NEAREST ) 
        ins = ins.resize( size, Image.NEAREST )
        return img, sem, ins 


    def crop(self, img, sem, ins, size):
        "size should be a tuple: W*H"
        img_width, img_height = img.size
        wanted_width, wanted_height = size

        width_start = random.randint(0, img_width-wanted_width-2) # for safe
        height_start = random.randint(0, img_height-wanted_height-2) # for safe
        
        width_end = width_start + wanted_width
        height_end = height_start + wanted_height
        
        img = img.crop( ( width_start, height_start, width_end, height_end ) )
        sem = sem.crop( ( width_start, height_start, width_end, height_end ) )
        ins = ins.crop( ( width_start, height_start, width_end, height_end ) )

        return img, sem, ins 


    def __getitem__(self, idx):
        
        # read raw images 
        img = Image.open( self.sourcer.img_bank[idx] ).convert('RGB')
        sem = Image.open( self.sourcer.sem_bank[idx] )
        ins = Image.open( self.sourcer.ins_bank[idx] )
                   
        
        if self.args.random_crop:
            # resize to bigger size
            size = ( int(self.args.scene_size[1]*1.2), int(self.args.scene_size[0]*1.2) )
            img, sem, ins = self.resize( img, sem, ins, size )
            # then crop
            size = ( self.args.scene_size[1], self.args.scene_size[0] )
            img, sem, ins = self.crop( img, sem, ins, size )

        else:
            # directly resize
            size = (self.args.scene_size[1], self.args.scene_size[0])
            img, sem, ins = self.resize(img, sem, ins, size)


        if self.args.random_flip and random.random()>0.5:
            img, sem, ins = self.flip(img, sem, ins)

        out = get_scene_data(img, sem, ins)
        
        return out
 

    def __len__(self):
        return len(self.sourcer.img_bank)
