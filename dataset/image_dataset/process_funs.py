#from dataset.instance_dataset.process_funs import process as instance_process
from dataset.utils.functions import *
import torch 
import torchvision.transforms.functional as TF
from torchvision import transforms    

from numpy import random

from trainer.render import render, set_param, getTexPos, height_to_normal

from trainer.utils import mycrop

import torchvision.transforms as T

# Convert [-1, 1] to [0, 1]
to_zero_one = lambda a: a / 2.0 + 0.5

# Convert to float tensor
to_tensor = lambda a: torch.as_tensor(a, dtype=torch.float)

def transform_2d(img_in, tile_mode=3, sample_mode='bilinear', mipmap_mode='auto', mipmap_level=0, x1=1.0, x1_max=1.0, x2=0.5, x2_max=1.0,
                 x_offset=0.5, x_offset_max=1.0, y1=0.5, y1_max=1.0, y2=1.0, y2_max=1.0, y_offset=0.5, y_offset_max=1.0,
                 ):
    """Atomic function: Transform 2D (https://docs.substance3d.com/sddoc/transformation-2d-172825332.html)

    Args:
        img_in (tensor): input image
        tile_mode (int, optional): 0=no tile, 
                                   1=horizontal tile, 
                                   2=vertical tile, 
                                   3=horizontal and vertical tile. Defaults to 3.
        sample_mode (str, optional): 'bilinear' or 'nearest'. Defaults to 'bilinear'.
        mipmap_mode (str, optional): 'auto' or 'manual'. Defaults to 'auto'.
        mipmap_level (int, optional): Mipmap level. Defaults to 0.
        x1 (float, optional): Entry in the affine transformation matrix, same for the below. Defaults to 1.0.
        x1_max (float, optional): . Defaults to 1.0.
        x2 (float, optional): . Defaults to 0.5.
        x2_max (float, optional): . Defaults to 1.0.
        x_offset (float, optional): . Defaults to 0.5.
        x_offset_max (float, optional): . Defaults to 1.0.
        y1 (float, optional): . Defaults to 0.5.
        y1_max (float, optional): . Defaults to 1.0.
        y2 (float, optional): . Defaults to 1.0.
        y2_max (float, optional): . Defaults to 1.0.
        y_offset (float, optional): . Defaults to 0.5.
        y_offset_max (float, optional): . Defaults to 1.0.
        matte_color (list, optional): background color. Defaults to [0.0, 0.0, 0.0, 1.0].

    Returns:
        Tensor: Transformed image.
    """
    assert sample_mode in ('bilinear', 'nearest')
    assert mipmap_mode in ('auto', 'manual')

    gs_padding_mode = 'zeros'
    gs_interp_mode = sample_mode

    x1 = to_tensor((x1 * 2.0 - 1.0) * x1_max).squeeze()
    x2 = to_tensor((x2 * 2.0 - 1.0) * x2_max).squeeze()
    x_offset = to_tensor((x_offset * 2.0 - 1.0) * x_offset_max).squeeze()
    y1 = to_tensor((y1 * 2.0 - 1.0) * y1_max).squeeze()
    y2 = to_tensor((y2 * 2.0 - 1.0) * y2_max).squeeze()
    y_offset = to_tensor((y_offset * 2.0 - 1.0) * y_offset_max).squeeze()

    # compute mipmap level
    mm_level = mipmap_level
    det = torch.abs(x1 * y2 - x2 * y1)
    if det < 1e-6:
        print('Warning: singular transformation matrix may lead to unexpected results.')
        mm_level = 0
    elif mipmap_mode == 'auto':
        inv_h1 = torch.sqrt(x2 * x2 + y2 * y2)
        inv_h2 = torch.sqrt(x1 * x1 + y1 * y1)
        max_compress_ratio = torch.max(inv_h1, inv_h2)
        # !! this is a hack !!
        upper_limit = 2895.329
        thresholds = to_tensor([upper_limit / (1 << i) for i in reversed(range(12))])
        mm_level = torch.sum(max_compress_ratio > thresholds).item()
        # Special cases
        is_pow2 = lambda x: torch.remainder(torch.log2(x), 1.0) == 0
        if torch.abs(x1) == torch.abs(y2) and x2 == 0 and y1 == 0 and is_pow2(torch.abs(x1)) or \
           torch.abs(x2) == torch.abs(y1) and x1 == 0 and y2 == 0 and is_pow2(torch.abs(x2)):
            scale = torch.max(torch.abs(x1), torch.abs(x2))
            if torch.remainder(x_offset * scale, 1.0) == 0 and torch.remainder(y_offset * scale, 1.0) == 0:
                mm_level = max(0, mm_level - 1)

    # mipmapping (optional)
    if mm_level > 0:
        mm_level = min(mm_level, int(np.floor(np.log2(img_in.shape[2]))))
        img_mm = automatic_resize(img_in, -mm_level)
        img_mm = manual_resize(img_mm, mm_level)
        assert img_mm.shape == img_in.shape
    else:
        img_mm = img_in

    # compute sampling tensor
    res_x, res_y = img_in.shape[3], img_in.shape[2]
    theta_first_row = torch.stack([x1, y1, x_offset * 2.0])
    theta_second_row = torch.stack([x2, y2, y_offset * 2.0])
    theta = torch.stack([theta_first_row, theta_second_row]).unsqueeze(0).expand(img_in.shape[0],2,3)
    sample_grid = torch.nn.functional.affine_grid(theta, img_in.shape, align_corners=False).cuda()

    sample_grid[:,:,:,0] = (torch.remainder(sample_grid[:,:,:,0] + 1.0, 2.0) - 1.0) * res_x / (res_x + 2)
    sample_grid[:,:,:,1] = (torch.remainder(sample_grid[:,:,:,1] + 1.0, 2.0) - 1.0) * res_y / (res_y + 2)


    pad_arr = [[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]]
    img_pad = torch.nn.functional.pad(img_mm, pad_arr[tile_mode], mode='circular')

    # compute output
    img_out = torch.nn.functional.grid_sample(img_pad, sample_grid, mode=gs_interp_mode, padding_mode=gs_padding_mode, align_corners=False)


    return img_out

def safe_transform(img_in, tile=1, tile_safe_rot=True, tile_mode=3, mipmap_mode='auto', mipmap_level=0, offset_x=0.0, offset_y=0.0, angle=0.2):
    """Non-atomic function: Safe Transform (https://docs.substance3d.com/sddoc/safe-transform-159450643.html)

    Args:
        img_in (tensor): Input image.
        tile (int, optional): Scales the input down by tiling it. Defaults to 1.
        tile_safe_rot (bool, optional): Determines the behaviors of the rotation, whether it should snap to 
            safe values that don't blur any pixels. Defaults to True.
        symmetry (str, optional): 'X'|'Y'|'X+Y'|'none', performs symmetric transformation on the input. Defaults to 'none'.
        tile_mode (int, optional): 0=no tile, 
                                   1=horizontal tile, 
                                   2=vertical tile, 
                                   3=horizontal and vertical tile. Defaults to 3.Defaults to 3.
        mipmap_mode (str, optional): 'auto' or 'manual'. Defaults to 'auto'.
        mipmap_level (int, optional): Mipmap level. Defaults to 0.
        offset_x (float, optional): x-axis offset. Defaults to 0.5.
        offset_y (float, optional): y-axis offset. Defaults to 0.5.
        angle (float, optional): Rotates input along angle. Defaults to 0.0.

    Returns:
        Tensor: Safe transformed image.
    """
    num_row = img_in.shape[2]
    num_col = img_in.shape[3]

    # main transform
    angle = to_tensor(angle)
    tile = to_tensor(tile)
    offset_tile = torch.remainder(tile + 1.0, 2.0) * to_tensor(0.5)
    if tile_safe_rot:
        angle = torch.floor(angle * 8.0) / 8.0
        angle_res = torch.remainder(torch.abs(angle), 0.25) * (np.pi * 2.0)
        tile = tile * (torch.cos(angle_res) + torch.sin(angle_res))
    offset_x = torch.floor((to_tensor(offset_x) * 2.0 - 1.0) * num_col) / num_col + offset_tile
    offset_y = torch.floor((to_tensor(offset_y) * 2.0 - 1.0) * num_row) / num_row + offset_tile
    # compute affine transformation matrix
    angle = angle * np.pi * 2.0
    scale_matrix = to_tensor([[torch.cos(angle), -torch.sin(angle)],[torch.sin(angle), torch.cos(angle)]])
    rotation_matrix = to_tensor([[tile, 0.0],[0.0, tile]])
    scale_rotation_matrix = torch.mm(rotation_matrix, scale_matrix)
    img_out = transform_2d(img_in, tile_mode=tile_mode, mipmap_mode=mipmap_mode, mipmap_level=mipmap_level,
                           x1=to_zero_one(scale_rotation_matrix[0,0]), x2=to_zero_one(scale_rotation_matrix[0,1]), x_offset=to_zero_one(offset_x), 
                           y1=to_zero_one(scale_rotation_matrix[1,0]), y2=to_zero_one(scale_rotation_matrix[1,1]), y_offset=to_zero_one(offset_y))
    return img_out


def automatic_resize(img_in, scale_log2, filtering='bilinear'):
    """Progressively resize an input image.

    Args:
        img_in (tensor): input image
        scale_log2 (int): size change relative to the input resolution (after log2)
        filtering (str, optional): 'bilinear' or 'nearest'. Defaults to 'bilinear'.

    Returns:
        Tensor: resized image
    """
    # Check input validity
    assert filtering in ('bilinear', 'nearest')
    in_size_log2 = int(np.log2(img_in.shape[2]))
    out_size_log2 = max(in_size_log2 + scale_log2, 0)

    # Equal size
    if out_size_log2 == in_size_log2:
        img_out = img_in
    # Down-sampling (regardless of filtering)
    elif out_size_log2 < in_size_log2:
        img_out = img_in
        for _ in range(in_size_log2 - out_size_log2):
            img_out = manual_resize(img_out, -1)
    # Up-sampling (progressive bilinear filtering)
    elif filtering == 'bilinear':
        img_out = img_in
        for _ in range(scale_log2):
            img_out = manual_resize(img_out, 1)
    # Up-sampling (nearest sampling)
    else:
        img_out = manual_resize(img_in, scale_log2, filtering)

    return img_out

def manual_resize(img_in, scale_log2, filtering='bilinear'):
    """Manually resize an input image (all-in-one sampling).

    Args:
        img_in (tensor): input image
        scale_log2 (int): size change relative to input (after log2).
        filtering (str, optional): 'bilinear' or 'nearest'. Defaults to 'bilinear'.

    Returns:
        Tensor: resized image
    """
    # Check input validity
    assert filtering in ('bilinear', 'nearest')
    in_size = img_in.shape[2]
    in_size_log2 = int(np.log2(in_size))
    out_size_log2 = max(in_size_log2 + scale_log2, 0)
    out_size = 1 << out_size_log2

    # Equal size
    if out_size_log2 == in_size_log2:
        img_out = img_in
    else:
        row_grid, col_grid = torch.meshgrid(torch.linspace(1, out_size * 2 - 1, out_size), torch.linspace(1, out_size * 2 - 1, out_size))
        sample_grid = torch.stack([col_grid, row_grid], 2).expand(img_in.shape[0], out_size, out_size, 2).cuda()
        sample_grid = sample_grid / (out_size * 2) * 2.0 - 1.0
        # Down-sampling
        if out_size_log2 < in_size_log2:
            img_out = torch.nn.functional.grid_sample(img_in, sample_grid, filtering, 'zeros', align_corners=False)
        # Up-sampling
        else:
            sample_grid = sample_grid * in_size / (in_size + 2)
            img_in_pad = torch.nn.functional.pad(img_in, (1, 1, 1, 1), mode='circular')
            img_out = torch.nn.functional.grid_sample(img_in_pad, sample_grid, filtering, 'zeros', align_corners=False)

    return img_out




def get_scene_data(args, img, composition=None):
    """
    This function is used by both background_dataset and refiner_dataset

    composition is also PIL.Image, but it is a generated image
    """
    full_img = TF.to_tensor(img).cuda() #[0,1]
    c,h,w = full_img.shape



    # data augmentation
    if args.aug_data:

        if args.dataset=='Leather':
            """
            if leather, we do tileable rotation + adjust hue,rough,height + crop
            """
            full_img = torch.cat([full_img[0:1,:,0:h],full_img[:,:,h:2*h],full_img[0:1,:,2*h:3*h],full_img[0:1,:,3*h:4*h]], dim=0).unsqueeze(0)
            full_img = safe_transform(full_img, angle = random.rand(), offset_x = random.rand(), offset_y = random.rand()).squeeze(0)

            gamma_hue = random.random()*0.2-0.1 # [-0.1~0.1]
            D = TF.adjust_hue(full_img[1:4,:,:], gamma_hue)

            gamma = 0.8+random.random()*0.4 # [0.8~1.2]
            H = full_img[0:1,:,:]**gamma
            R = full_img[4:5,:,:]**gamma

            rand0 = [random.randint(-args.scene_size[0]+1, D.shape[-1]-1),random.randint(-args.scene_size[0]+1, D.shape[-1]-1)]

            # randomly crop
            scene_sem = full_img[5:6,:,:]
            scene_img = torch.cat([H, D, R], dim=0)
            scene_sem = mycrop(scene_sem.unsqueeze(0), H.shape[-1], rand0=rand0).squeeze(0)
            scene_img = mycrop(scene_img.unsqueeze(0), H.shape[-1], rand0=rand0).squeeze(0)

        elif args.dataset=='Stone':

            from scipy.ndimage import gaussian_filter
            """
            if stone, we do blur + tileable rotation + adjust hue,rough,height + crop
            """
            # print(full_img[0,:,0:h].shape)
            P = gaussian_filter(full_img[0,:,0:h].cpu().numpy(), sigma=random.randint(8, 12), mode='wrap')
            # print(P.shape)

            minv = np.amin(P)
            maxv = np.amax(P) 

            if maxv-minv > 0.1:
                P =(P - minv)/(maxv-minv)
                P = 1/(1 + np.exp(-(P-0.5)*5))
            else:
                P = P*0+1

            P = TF.to_tensor(P).cuda()
            # print(P.shape)    

            full_img = torch.cat([full_img[0:1,:,0:h],full_img[:,:,h:2*h],full_img[0:1,:,2*h:3*h],P], dim=0).unsqueeze(0)
            full_img = safe_transform(full_img, angle = random.rand(), offset_x = random.rand(), offset_y = random.rand()).squeeze(0)

            gamma_hue = random.random()*0.2-0.1 # [-0.1~0.1]
            D = TF.adjust_hue(full_img[1:4,:,:], gamma_hue)

            gamma = 0.8+random.random()*0.4 # [0.8~1.2]
            H = full_img[0:1,:,:]**gamma
            R = full_img[4:5,:,:]**gamma

            rand0 = [random.randint(-args.scene_size[0]+1, D.shape[-1]-1),random.randint(-args.scene_size[0]+1, D.shape[-1]-1)]

            # randomly crop
            scene_sem = full_img[5:6,:,:]
            scene_img = torch.cat([H, D, R], dim=0)
            scene_sem = mycrop(scene_sem.unsqueeze(0), H.shape[-1], rand0=rand0).squeeze(0)
            scene_img = mycrop(scene_img.unsqueeze(0), H.shape[-1], rand0=rand0).squeeze(0)

        elif args.dataset=='Tile':

            gamma_hue = random.random()*0.1-0.05 # [-0.1~0.1]
            D = TF.adjust_hue(full_img[:,:,h:2*h], gamma_hue)

            gamma = 0.9+random.random()*0.2 # [0.8~1.2]
            H = full_img[0:1,:,0:h]**gamma
            R = full_img[0:1,:,2*h:3*h]**gamma

            scene_img = torch.cat([H, D, R], dim=0)

            if args.color_cond:
                # scene_sem = full_img[0:1,:,4*h:5*h]
                # scene_sem = color_jitter(full_img[:,:,4*h:5*h]).cuda()
                scene_sem = TF.adjust_hue(full_img[:,:,4*h:5*h], gamma_hue)
            else:
                scene_sem = full_img[0:1,:,3*h:4*h]
            
            rand0 = [random.randint(-args.scene_size[0]+1, D.shape[-1]-1),random.randint(-args.scene_size[0]+1, D.shape[-1]-1)]

            # randomly crop
            # print('before: ',scene_img.shape)
            scene_sem = mycrop(scene_sem.unsqueeze(0), H.shape[-1], rand0=rand0).squeeze(0)
            scene_img = mycrop(scene_img.unsqueeze(0), H.shape[-1], rand0=rand0).squeeze(0)
            # print('after: ',scene_img.shape)


        elif args.dataset=='Metal':

            # from scipy.ndimage import gaussian_filter
            """
            if stone, we do blur + tileable rotation + adjust hue,rough,height + crop
            """
            # print(full_img[0,:,0:h].shape)
            # P = gaussian_filter(full_img[0,:,0:h].cpu().numpy(), sigma=random.randint(8, 12), mode='wrap')
            # print(P.shape)

            # minv = np.amin(P)
            # maxv = np.amax(P) 

            # if maxv-minv > 0.1:
            #     P =(P - minv)/(maxv-minv)
            #     P = 1/(1 + np.exp(-(P-0.5)*5))
            # else:
            #     P = P*0+1

            # P = TF.to_tensor(P).cuda()
            # print(P.shape)    

            full_img = torch.cat([full_img[0:1,:,0:h],full_img[:,:,h:2*h],full_img[0:1,:,2*h:3*h],full_img[0:1,:,3*h:4*h]], dim=0).unsqueeze(0)
            full_img = safe_transform(full_img, angle = random.rand(), offset_x = random.rand(), offset_y = random.rand()).squeeze(0)

            gamma_hue = random.random()*0.2-0.1 # [-0.1~0.1]
            D = TF.adjust_hue(full_img[1:4,:,:], gamma_hue)

            gamma = 0.9+random.random()*0.2 # [0.9~1.1]
            H = full_img[0:1,:,:]**gamma
            R = full_img[4:5,:,:]**gamma

            M = full_img[5:6,:,:]

            rand0 = [random.randint(-args.scene_size[0]+1, D.shape[-1]-1),random.randint(-args.scene_size[0]+1, D.shape[-1]-1)]

            # randomly crop
            scene_sem = full_img[5:6,:,:]
            scene_img = torch.cat([H, D, R, M], dim=0)

            scene_sem = mycrop(scene_sem.unsqueeze(0), H.shape[-1], rand0=rand0).squeeze(0)
            scene_img = mycrop(scene_img.unsqueeze(0), H.shape[-1], rand0=rand0).squeeze(0)


    else:
        H = full_img[0:1,:,0:h]
        D = full_img[:,:,h:2*h]
        R = full_img[0:1,:,2*h:3*h]
        scene_img = torch.cat([H, D, R], dim=0)

    # extract conditional mask
    out = {}

    out['scene_sem'] = 2*scene_sem-1 if not args.scalar_cond else 2*scene_img-1
    out['scene_img'] = 2*scene_img-1

    # print('out[scene_img] shape before: ', out['scene_img'].shape)

    if out['scene_img'].shape[-1]!=args.scene_size[0]:
        import torch.nn.functional as F

        out['scene_sem'] = F.interpolate(out['scene_sem'].unsqueeze(0), size=(args.scene_size[0], args.scene_size[1]), mode='bilinear').squeeze(0)
        out['scene_img'] = F.interpolate(out['scene_img'].unsqueeze(0), size=(args.scene_size[0], args.scene_size[1]), mode='bilinear').squeeze(0)

    return out




# def process(args, img, sem, ins, parsing):
#     """   

#     This is the entrance of processing used by composition_dataset.py

#     img: PIL.Image with the original shape 
#     sem: PIL.Image with the original shape
#     ins: PIL.Image with the original shape
#     parsing: a refined parsing (a dict)
    
#     It will return 
#     bg_data: a dict containing data needed for background generator 
#     fg_data: a list containing multiple dict which stores data needed for foreground generator   
#     """
    
#     bg_data = get_scene_data(img, sem, ins, parsing=parsing)    
#     fg_data = get_foreground_data(args, img, sem, ins, parsing)    
#     return bg_data, fg_data










################          EXPLANTION_1            ################## 
#
# remember that these those images are resized to scene_size
# thus potentially it may remove small instances, 
# while our parsing is derived from the orignal resolution.
#
#######################################################################