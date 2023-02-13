from config.background_cfg import BackgroundConfig
import torch 
import pandas as pd 
from dataset.utils.functions import *
from dataset.image_dataset.datasourcer import DataSourcer
from PIL import Image 
from copy import deepcopy
from dataset.image_dataset.process_funs import get_background_data
import random
import torchvision.transforms.functional as TF

def refine_parsing(parsing):
    """
    Here we will refine its order.
    Order in parsing is a list containing all instances from big to small.
    This function will try to remove those background instace.    
    Purpose: then you can use refined order to decide which instances are not visible,
    otherwise you can not do that because unrefined order contains background instances
    """
    
    # get old order and order_name, and location of fg instance     
    order = parsing['order']
    order_name = parsing['order_name']
    keep_idxs = [i for i, name in enumerate(order_name)  if name in BackgroundConfig.FG_CLASSES   ]
            
    # keep fg instance (remove bg instance)
    order = [item for i, item in enumerate(order) if i in keep_idxs ]
    order_name = [item for i, item in enumerate(order_name) if i in keep_idxs ]
    
    # rewrite order and order_name
    parsing['order'] = order  
    parsing['order_name'] = order_name

    return parsing





class Dataset(torch.utils.data.Dataset):
    "This is a regular dataset class"
    def __init__(self, args, train):
        self.args = args 
        self.train = train 

        # first create mapping 
        data = pd.read_csv( BackgroundConfig.OBJ150_PATH, sep='\t', lineterminator='\n') 
        mapping = {}
        for i in range(150):
            line = data.loc[i]
            mapping[ line['Name']  ] = line['Idx']
            
        # prepare datasourcer  
        names = BackgroundConfig.TRAIN_DATASETS if train else BackgroundConfig.TEST_DATASETS
        paths_list = get_path(names, train=train)
        self.sourcer = DataSourcer(paths_list)    
        
        print('total data: ', len(self.sourcer.img_bank) )


    def flip(self, img, sem, ins):
        return  TF.hflip(img), TF.hflip(sem), TF.hflip(ins)
        

    def __getitem__(self, idx):
        
        # read raw images and parsing 
        img = Image.open( self.sourcer.img_bank[idx] ).convert('RGB')
        sem = Image.open( self.sourcer.sem_bank[idx] )
        ins = Image.open( self.sourcer.ins_bank[idx] )
        parsing = refine_parsing( deepcopy(self.sourcer.parsing_bank[idx]) )
                
        # resize into args.scene_size; 
        img = img.resize( (self.args.scene_size,self.args.scene_size), Image.NEAREST ) 
        sem = sem.resize( (self.args.scene_size,self.args.scene_size), Image.NEAREST ) 
        ins = ins.resize( (self.args.scene_size,self.args.scene_size), Image.NEAREST ) 

        if self.args.random_flip and random.random()>0.5:
            img, sem, ins = self.flip(img, sem, ins)

        out = get_background_data(img, sem, ins, parsing)
        scene_img = out['scene_img'].squeeze(0)
        scene_sem = out['scene_sem'].squeeze(0)
        scene_ins = out['scene_ins'].squeeze(0)
        scene_seg = out['scene_seg'].squeeze(0)
 
        return scene_img, scene_sem, scene_ins, scene_seg
 

    def __len__(self):
        return len(self.sourcer.img_bank)
