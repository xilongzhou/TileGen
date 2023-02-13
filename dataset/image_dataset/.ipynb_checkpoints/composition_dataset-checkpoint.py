from config.composition_cfg import CompositionConfig
import torch 
import pandas as pd 
from dataset.utils.functions import *
from dataset.image_dataset.datasourcer import DataSourcer
from PIL import Image 
from copy import deepcopy





class Dataset(torch.utils.data.Dataset):
    "This is a regular dataset class"
    def __init__(self, args):
        self.args = args 
               
        assert not args.random_flip
            
        # prepare datasourcer  
        paths = get_path(CompositionConfig.DATASET, train=args.use_train, load_generated=args.load_generated)
        self.sourcer = DataSourcer( [paths] )    
        

    def __getitem__(self, idx):
        """
        For composition, we only need to provide the very first starting image
        As for the later data needed by instance-level generator, it will call dataset/instance_dataset/dataset.py 
        The reason why we provide idx is that this is global_index of this image, we need it to create instance_dataset        
        """
        img = Image.open( self.sourcer.img_bank[idx] ).convert('RGB')
        img = img.resize( (self.args.scene_size[1],self.args.scene_size[0]), Image.NEAREST ) 
        return img, idx
 

    def __len__(self):
        return len(self.sourcer.img_bank)
