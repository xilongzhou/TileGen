import shutil
import glob
import torch.nn as nn
import torch 
from torchvision import  utils
import os 
import torch.nn.functional as F
import PIL
import numpy as np
from PIL import Image 

import random

# self-defined crop function
def mycrop(x, size, center=False, rand0=None, tileable=True):

    b,c,h,w = x.shape
    if center:
        w0 = (w - size)*0.5
        h0 = (h - size)*0.5
        w0 = int(w0)
        h0 = int(h0)

        # print('center: ', w0, h0)
        if w0+size>w or h0+size>h:
            raise ValueError('value error of w0')

        return x[:,:,w0:w0+size,h0:h0+size]

    if not tileable:
        if rand0 is None:
            w0 = random.randint(0,w-size)
            h0 = random.randint(0,h-size)
        else:
            h0 = rand0[0]
            w0 = rand0[1]

        if w0+size>w or h0+size>h:
            raise ValueError('value error of w0')
        print('rand: ', w0, h0)

        return x[:,:,w0:w0+size,h0:h0+size]

    else:
        if rand0 is None:
            w0 = random.randint(-size+1,w-1)
            h0 = random.randint(-size+1,h-1)
        else:
            h0 = rand0[0]
            w0 = rand0[1]

        wc = w0 + size
        hc = h0 + size

        p = torch.ones((b,c,size,size), device='cuda')

        # seperate crop and stitch them manually
        # [7 | 8 | 9]
        # [4 | 5 | 6]
        # [1 | 2 | 3]
        # 1
        if h0<=0 and w0<=0:
            p[:,:,0:-h0,0:-w0] = x[:,:, h+h0:h, w+w0:w]
            p[:,:,-h0:,0:-w0] = x[:,:, 0:hc, w+w0:w]
            p[:,:,0:-h0,-w0:] = x[:,:, h+h0:h, 0:wc]
            p[:,:,-h0:,-w0:] = x[:,:, 0:hc, 0:wc]
        # 2
        elif h0<=0 and (w0<w-size and w0>0):
            p[:,:,0:-h0,:] = x[:,:, h+h0:h,w0:wc]
            p[:,:,-h0:,:] = x[:,:, 0:hc, w0:wc]
        # 3
        elif h0<=0 and w0 >=w-size:
            p[:,:,0:-h0,0:w-w0] = x[:,:, h+h0:h, w0:w]
            p[:,:,-h0:,0:w-w0] = x[:,:, 0:hc, w0:w]
            p[:,:,0:-h0,w-w0:] = x[:,:, h+h0:h, 0:wc-w]
            p[:,:,-h0:,w-w0:] = x[:,:, 0:hc, 0:wc-w]

        # 4
        elif (h0>0 and h0<h-size) and w0<=0:
            p[:,:,:,0:-w0] = x[:,:, h0:hc, w+w0:w]
            p[:,:,:,-w0:] = x[:,:, h0:hc, 0:wc]
        # 5
        elif (h0>0 and h0<h-size) and (w0<w-size and w0>0):
            p = x[:,:, h0:hc, w0:wc]
        # 6
        elif (h0>0 and h0<h-size) and w0 >=w-size:
            p[:,:,:,0:w-w0] = x[:,:, h0:hc, w0:w]
            p[:,:,:,w-w0:] = x[:,:, h0:hc, 0:wc-w]

        # 7
        elif h0 >=h-size and w0<=0:
            p[:,:,0:h-h0,0:-w0] = x[:,:, h0:h, w+w0:w]
            p[:,:,h-h0:,0:-w0] = x[:,:, 0:hc-h, w+w0:w]
            p[:,:,0:h-h0,-w0:] = x[:,:, h0:h, 0:wc]
            p[:,:,h-h0:,-w0:] = x[:,:, 0:hc-h, 0:wc]
        # 8
        elif h0 >=h-size and (w0<w-size and w0>0):
            p[:,:,0:h-h0,:] = x[:,:, h0:h,w0:wc]
            p[:,:,h-h0:,:] = x[:,:, 0:hc-h, w0:wc]
        # 9
        elif h0 >=h-size and w0 >=w-size:
            p[:,:,0:h-h0,0:w-w0] = x[:,:, h0:h, w0:w]
            p[:,:,h-h0:,0:w-w0] = x[:,:, 0:hc-h, w0:w]
            p[:,:,0:h-h0,w-w0:] = x[:,:, h0:h, 0:wc-w]
            p[:,:,h-h0:,w-w0:] = x[:,:, 0:hc-h, 0:wc-w]

        del x

        return p




def sample_data(args, loader):
    epoch = 0 
    while True:
        if args.distributed:
            loader.sampler.set_epoch(epoch) # otherwise, each time you will have the same order for all epoch
        epoch += 1
        for batch in loader:
            yield batch



class ImageSaver():
    def __init__(self, base_path, nrow=1, normalize=True, range=(-1,1) ):
        self.base_path = base_path 
        self.nrow = nrow
        self.normalize = normalize
        self.range = range

    def __call__(self, x, name, gamma=False):
        "x: Tensor or PIL.Image. name: a str"
        # print(name, x.shape)

        if x.shape[1]==5:
            x = torch.cat([x[:,0:1,:,:].repeat(1,3,1,1), 2*((x[:,1:4,:,:]+1)*0.5)**(1/2.2)-1, x[:,4:5,:,:].repeat(1,3,1,1)],dim=-1)
        elif x.shape[1]==6:
            x = torch.cat([x[:,0:1,:,:].repeat(1,3,1,1), 2*((x[:,1:4,:,:]+1)*0.5)**(1/2.2)-1, x[:,4:5,:,:].repeat(1,3,1,1), x[:,5:6,:,:].repeat(1,3,1,1)],dim=-1)
            # x = torch.cat([x[:,0:1,:,:].repeat(1,3,1,1), x[:,1:4,:,:], x[:,4:5,:,:].repeat(1,3,1,1), x[:,5:6,:,:].repeat(1,3,1,1)],dim=-1)
        else:
            if gamma:
                x = 2*((x+1)*0.5)**(1/2.2)-1

        save_path = os.path.join(self.base_path, name)
        if type(x) == torch.Tensor: 
            utils.save_image( x, save_path, nrow=self.nrow, normalize=self.normalize, range=self.range )
        elif type(x) == PIL.Image.Image:
            x.save(save_path)
           


class CheckpointSaver():
    def __init__(self, args,  base_path ):
        """
        we will keep saving ckpt after every opt.ckpt_save_frenquency. However, before hitting opt.start_keeping_iter
        once a new ckpt is saved, old one will be removed for saving space. But after opt.start_keeping_iter, 
        all saved ckpt will be kept. 
        """
        self.base_path = base_path 
        self.start_keeping_iter = args.start_keeping_iter
        self.ckpt_save_frenquency = args.ckpt_save_frenquency

    def __call__(self, ckpt, count):

        save_path = os.path.join( self.base_path, str(count).zfill(6)+'.pt' )
        torch.save(ckpt, save_path)

        old_ckpt = os.path.join( self.base_path, str(count-self.ckpt_save_frenquency).zfill(6)+'.pt' )
        if os.path.exists(old_ckpt) and count <= self.start_keeping_iter:
            os.remove(old_ckpt)




def to_device(input, device):
    "input is a dict, we will seach all items in this dict and send to device if it is tensor"
    for key in input:
        if type(input[key]) == torch.Tensor:
            input[key] = input[key].to(device)
    return input 



def sample_n_data(num_sample, loader, current_batch):
    """
    num_sample is an int value
    loader: pytorch dataloader class wrapped by sample_data()
    current_batch: batch_size of loader 

    This function will return a data same as output of dataloader, but its batchsize is euqal to num_sample 
    You can think as you re-define batchsize of the input loader and output an one sample.

    For now output of loader must be a dict, and all items in it must be tensor
    This function looks stupid, do not call it multiple times 
    """

    def helper(output, keys):
        for key in keys:
            if key not in output:
                output[key] = [] 
        return output
 
    
    output = {}
    count = 0 
    while True:        
        data = next(loader)  
        output = helper( output, list(data.keys()) )   
        for i in range(  current_batch ):   
            if count == num_sample: # break two loops here
                break
            for key in data:
                output[key].append(  data[key][i]  )    
            count+=1
        else:
            continue  
        break  

    # then stack all 
    for key in data:
        output[key] = torch.stack( output[key]  )
    
    return output




def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)






def random_colors(N, seed=None):
    "Generate random colors. Generate them in HSV space then convert to RGB."
    if seed:
        np.random.seed(seed)
    r = np.random.random_sample(151)
    g = np.random.random_sample(151)
    b = np.random.random_sample(151)

    return list(zip(r,g,b))




class SemanticMapVisualizer:
    def __init__(self, total_num, start_from_zero):
        """
        total_num: all possible different class labels (including zero if you have)
        start_from_zero: if start from zero or not
       
        Note that it only supports two continuous cases
        case1: 0,1,2,3,4,....
        case2: 1,2,3,4..
       
        You can not label your class like this: 0,1,2,4,5
        """
        if total_num == 151:
            seed = 1 # ade color space 
        elif total_num == 34:
            seed = 8 # cityscapes 
        elif total_num == 24:
            seed = 9 # human 
        else:
            seed = 1 # other default
        colors = random_colors( total_num,  seed=seed )
       
        self.map = {}
        start = 0 if start_from_zero else 1
        for idx, label in enumerate(range(start, start+total_num)):
            self.map[label] = colors[idx]
   
    def __call__(self, x):
        "input x should be torch.tensor, it will return PIL Image"
       

        x = x.squeeze()
        assert x.ndim == 2, 'weird input semantic map shape after squeeze'
       
        out_r = torch.zeros_like(x).float()
        out_g = torch.zeros_like(x).float()
        out_b = torch.zeros_like(x).float()
       
        labels = torch.unique(x)
        for label in labels:
            color = self.map[int(label)]
            out_r.masked_fill_(  x==label, color[0] )
            out_g.masked_fill_(  x==label, color[1] )
            out_b.masked_fill_(  x==label, color[2] )
           
        out = torch.stack(  [out_r,out_g,out_b]  )
       
        return  Image.fromarray((np.array(out).transpose(1,2,0)*255).astype('uint8'))






class CodeDependency():
    def __init__(self, CompositionConfig):
        
        self.groups = {}
        self.all_members = []
                
        for class_dict in CompositionConfig.PRETRAINED_MODELS:
            if 'code_dependency_idx' in class_dict.keys():
                
                class_name = list(class_dict.keys())[0]
                self.all_members.append(class_name)
                group_idx = class_dict['code_dependency_idx']
                
                if group_idx not in self.groups:
                    group_info = { 'group_member':[], 'z_code':None   }
                    self.groups[group_idx] = group_info
                    
                self.groups[group_idx]['group_member'].append( class_name )
        
        # now check if any group only has one member 
        for k in self.groups:
            assert len(  self.groups[k]['group_member']  ) > 1, 'as least two classes should have the same code_dependency_idx' 

    def check_dependency(self, class_name):
        "To see if input class is in any group"
        if class_name in self.all_members:
            return True
        else:
            return False            

    def which_group(self, class_name):
        for k in self.groups:
            if class_name in self.groups[k]['group_member']:
                return k        
        return None


    def store_code(self, class_name, code):
        "input class_name should be in a group"
        assert class_name in self.all_members
        
        group_info = self.groups[  self.which_group(class_name)  ]
        group_info['z_code'] = code

    
    def get_code(self, class_name):
        "input class_name should be in a group"
        assert class_name in self.all_members
        
        group_info = self.groups[  self.which_group(class_name)  ]
        return group_info['z_code']