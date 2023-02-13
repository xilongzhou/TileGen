import numpy as np
import random
import os 


    
def modify_x2y2(x2, y2):
    return x2+1, y2+1 
    

def exist_check(x1, y1, x2, y2):
    if x1 == y1 ==x2 == y2 == 0:
        return False 
    else:
        return True
    
        
def enlarge_box(x1, y1, x2, y2, width, height, ratio):
    w, h = x2-x1, y2-y1
    r = int( max(w,h) * (ratio/2) )
    center_x = int( (x1+x2)/2 )
    center_y = int( (y1+y2)/2 )
    y1 = max(0, center_y-r)
    y2 = min(height, center_y+r)
    x1 = max(0, center_x-r)
    x2 = min(width, center_x+r)
    return x1, y1, x2, y2





def get_box(mask):
    "mask should be a 2D np.array " 
    y,x = np.where(mask == 1)
    try:
        x1,x2,y1,y2 = x.min(),x.max(),y.min(),y.max()
    except ValueError:
        x1 = x2= y1 = y2 = 0        

    return x1,y1,x2,y2

        
        
def random_shift_box(x1, y1, x2, y2, width, height, ratio):
    # if ratio=0.2 then shift range is from -0.2*half_w to 0.2*half_w 
    half_w, half_h = int((x2-x1)/2), int((y2-y1)/2)
    center_x = int( (x1+x2)/2 ) + random.randint( int(-ratio*half_w), int(ratio*half_w)  )
    center_y = int( (y1+y2)/2 ) + random.randint( int(-ratio*half_h), int(ratio*half_h)  )

    y1 = max(0, center_y-half_h)
    y2 = min(height, center_y+half_h)
    x1 = max(0, center_x-half_w)
    x2 = min(width, center_x+half_w)
    return x1, y1, x2, y2





# def refine_parsing(parsing):
#     """
#     Here we will refine its order.
#     Order in parsing is a list containing all instances from big to small.
#     This function will try to remove those background instace.    
#     Purpose: then you can use refined order to decide which instances are not visible,
#     otherwise you can not do that because unrefined order contains background instances
#     """
    
#     # get old order and order_name, and location of fg instance     
#     order = parsing['order']
#     order_name = parsing['order_name']
#     keep_idxs = [i for i, name in enumerate(order_name)  if name in BaseConfig.FG_CLASSES   ]
            
#     # keep fg instance (remove bg instance)
#     order = [item for i, item in enumerate(order) if i in keep_idxs ]
#     order_name = [item for i, item in enumerate(order_name) if i in keep_idxs ]
    
#     # rewrite order and order_name
#     parsing['order'] = order  
#     parsing['order_name'] = order_name

#     return parsing





def check_and_return(path):
    assert os.path.exists(path), path+' not exists'
    return path 



def get_path( name, class_name=None, train=True, load_generated=False, load_composition=False ):
    """
    
    name should be path to a dataset root
    
    All dataset root folder should follow the same structure:
        full_data:
            images
            annotations
            annotations_instance 
            generated (optional)
            composition (optional)
        class1_info
        class2_info
        ...
    
    It will return a path(dict) used by dataset 
                
    """
    
    temp = 'training' if train else 'validation'

    path = {} 

    if load_generated:
        path['img'] = check_and_return( os.path.join(name,'full_data','generated',temp)  )
    else:
        path['img'] = check_and_return( os.path.join(name,'full_data','images',temp)  )
    path['sem'] = check_and_return( os.path.join(name,'full_data','annotations',temp ) )
    path['ins'] = check_and_return( os.path.join(name,'full_data','annotations_instance',temp)  )        

    if class_name:
        path['info'] = check_and_return( os.path.join(name,class_name,temp+'_info.pkl')  )
        # if sampling multiplier exists, load it 
 
        multiplier_path = os.path.join(name,class_name,temp+'_multiplier.pkl')
        if os.path.exists( multiplier_path ):
            path['multiplier'] = multiplier_path
         
    if load_composition:
        path['composition'] = check_and_return( os.path.join(name,'full_data','composition',temp)  )

    
    return path 
        





