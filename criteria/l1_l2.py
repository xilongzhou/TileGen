import torch.nn as nn



class L1_loss():
    def __init__(self):
        self.L1 = nn.L1Loss(reduction='none')
        self.L1_mean = nn.L1Loss()
    def __call__(self, x, y, mask=None):
        
        if mask is None:
            return self.L1_mean(x,y)
        else:
            num_c = x.shape[1]
            loss = self.L1(x,y)*mask
            loss = loss.sum() / (mask.sum()*num_c)
            return loss 


class L2_loss():
    def __init__(self):
        self.L2 = nn.MSELoss(reduction='none')
        self.L2_mean = nn.MSELoss()
    def __call__(self, x, y, mask=None):
        
        if mask is None:
            return self.L2_mean(x,y)
        else:
            num_c = x.shape[1]
            loss = self.L2(x,y)*mask
            loss = loss.sum() / (mask.sum()*num_c)
            return loss 