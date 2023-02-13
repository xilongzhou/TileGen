from torch import autograd
import torch.nn.functional as F
import torch 
import math



EPSILON=1e-6

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    """
    explain this loss:
        autograd will calculate gradient of output with respect to each input value.
        For example if you input is torch.tensor([1,2,3]) and you do out=(input*torch.tensor([5])).sum()
        Then your grad_real will be torch.tensor([5,5,5]). Basically measure if you change 1 in input space
        how much it will affact output 

        Here grad_real is a gradient tensor with the size N*3*H*W, later it is reshaped into N*(3*H*W) to
        calculate the loss    
    """
    grad_real, = autograd.grad( outputs=real_pred.sum(), inputs=real_img, create_graph=True )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty




def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):

    # latents is bs*num_layers*512
    noise = torch.randn_like(fake_img) / math.sqrt( fake_img.shape[2] * fake_img.shape[3] )
    grad, = autograd.grad( outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1)+EPSILON)
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_penalty = (path_lengths - path_mean).pow(2).mean()
    return path_penalty, path_mean.detach()


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss