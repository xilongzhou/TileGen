from dataset.instance_dataset.process_funs import process as instance_process
from dataset.utils.functions import *
import torch 
import torchvision.transforms.functional as TF


def get_foreground_data(args, img, sem, ins, parsing):    
    output = []    
    order = parsing['order']
    order_name = parsing['order_name']
    for ins_idx, class_name in zip(order, order_name):        
        box = get_box( np.array(ins)==ins_idx )        
        if exist_check(*box):  # see EXPLANTION_1 
            not_shown_inss = order[order.index(ins_idx):]
            out = to_dict( instance_process(args, img, sem, ins, ins_idx, box, not_shown_inss, mode='image')  )
            out['class_name'] = class_name
            output.append(out)
            
    return output




def to_dict(datum):
    target_seg, global_sem, global_pri, global_seg, composition_seg, info  = datum     
    out={}    
    out['target_seg'] = target_seg.unsqueeze(0) # used in Foreground Generator
    out['global_sem'] = global_sem.unsqueeze(0) # used in Encoder
    out['global_pri'] = global_pri.unsqueeze(0) # used in Encoder
    out['global_seg'] = global_seg.unsqueeze(0) # used in Encoder
    out['composition_seg'] = composition_seg.unsqueeze(0) # used in composition 
    out['info'] = info    
    return out 



def get_background_data(img, sem, ins, composition=None, parsing=None):
    """
    This function is used by background_dataset and refiner_dataset

    composition is also PIL.Image, but it is a generated image
    parsing here is only used to generate binary mask indicating bg vs fg 
    """

    scene_img = ( TF.to_tensor(img) - 0.5 ) / 0.5 
    scene_sem = torch.tensor( np.array(sem) ).unsqueeze(0).long()
    scene_ins = torch.tensor( np.array(ins) ).unsqueeze(0).long()
    
    if parsing != None:
        scene_seg = torch.ones_like(scene_ins).float()    
        for idx in parsing['order']: # parsing should be refined beforehand, thus order only contains fg instance 
            scene_seg[scene_ins==idx] = 0
    
    if composition != None:
        composition = ( TF.to_tensor(composition) - 0.5 ) / 0.5 


    out = {}
    out['scene_img'] = scene_img.unsqueeze(0)
    out['scene_sem'] = scene_sem.unsqueeze(0)
    out['scene_ins'] = scene_ins.unsqueeze(0)
    if parsing != None:
        out['scene_seg'] = scene_seg.unsqueeze(0)
    if composition != None:
        out['composition'] = composition.unsqueeze(0)

    return out




def process(args, img, sem, ins, parsing):
    """   

    This is the entrance of processing used by composition_dataset.py

    img: PIL.Image with the original shape 
    sem: PIL.Image with the original shape
    ins: PIL.Image with the original shape
    parsing: a refined parsing (a dict)
    
    It will return 
    bg_data: a dict containing data needed for background generator 
    fg_data: a list containing multiple dict which stores data needed for foreground generator   
    """
    
    bg_data = get_background_data(img, sem, ins, parsing=parsing)    
    fg_data = get_foreground_data(args, img, sem, ins, parsing)    
    return bg_data, fg_data










################          EXPLANTION_1            ################## 
#
# remember that these those images are resized to scene_size
# thus potentially it may remove small instances, 
# while our parsing is derived from the orignal resolution.
#
#######################################################################