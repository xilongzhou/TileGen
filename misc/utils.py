import shutil
import glob
import torch.nn as nn
import torch 


def save_all_training_files(output_dir):
    
    # save all py files 
    all_python_files = glob.glob("./*.py")
    for file in all_python_files:
        shutil.copy2(file, output_dir)





class RGB_Enlarger(nn.Module):
    def __init__(self):
        super().__init__()
        self.hard_enlarger = PartialConv()
 
    def __call__(self, x, mask):
        "For now it is hardcode applying once hard enlarger"
    
        # hard expand by 2 pixels (inside will be blur) 
        enlarged_x = self.hard_enlarger(x, mask)
        # restore inside image 
        x = enlarged_x*(1-mask) + x*mask
        
        return x 


class Mask_Enlarger(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,1,3,1,1, bias=False)
        self.conv.weight.data.fill_(1/9)
        
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = self.conv(x) *x         
        return x 





class PartialConv(nn.Module):
    def __init__(self):
        super().__init__()

               
        self.mask_conv = nn.Conv2d(1, 1, 3, 1, 1, bias=False)        
        self.mask_conv.weight.data.fill_(1.0)
        
        self.input_conv = nn.Conv2d(3, 3, 3, 1, 1, bias=False) 
        for i in range(3):
            init = torch.zeros(3,3,3)
            init[i,:,:] = 1/9 
            self.input_conv.weight.data[i] = init
                   
        for param in self.parameters():
            param.requires_grad = False
            
 
    def forward(self, input, mask):

        output = self.input_conv( input*mask )
        mask = self.mask_conv(mask)

        no_update_holes = mask == 0
        mask_ratio = (3*3) / mask.masked_fill_(no_update_holes, 1.0)

        output = output * mask_ratio 
        output = output.masked_fill_(no_update_holes, 0.0)
 
        return output


