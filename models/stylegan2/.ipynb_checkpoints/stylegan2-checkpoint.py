import math
import random
import torch
from torch import nn
from torch.nn import functional as F


from models.stylegan2.building_blocks import PixelNorm, EqualLinear, ConstantInput, StyledConv, ConvLayer, ResBlock, ToRGB 




class Generator(nn.Module):
    def __init__(self, size, style_dim, n_mlp, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01, ):
        super().__init__()

        self.size = size
        self.style_dim = style_dim


        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append( EqualLinear( style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu' )  )
        self.style = nn.Sequential(*layers)


        self.channels = {   4: 512,
                            8: 512,
                            16: 512,
                            32: 512,
                            64: 256 * channel_multiplier,
                            128: 128 * channel_multiplier,
                            256: 64 * channel_multiplier,
                            512: 32 * channel_multiplier,
                            1024: 16 * channel_multiplier }


        self.input = ConstantInput(self.channels[4]) # a learnable constance 1*512*4*4 
        self.conv1 = StyledConv( self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.n_latent = self.log_size * 2 - 2

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))


        in_channel = self.channels[4]
        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append( StyledConv( in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel ) )
            self.convs.append( StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel) )
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel

        

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn( n_latent, self.style_dim, device=self.input.input.device)
        latent = self.style(latent_in).mean(0, keepdim=True).unsqueeze(1)
        return latent

    def get_latent(self, input):
        return self.style(input)




    def __prepare_letent(self, styles, inject_index, truncation, truncation_latent,  input_type):
        "This is a private function to prepare w+ space code needed during forward"

        if input_type == 'z':
            styles = [self.style(s).unsqueeze(1) for s in styles]  # each one is bs*1*512
        elif input_type == 'w':
            styles = [s.unsqueeze(1) for s in styles]  # each one is bs*1*512
        else: 
            return styles # w+ case 


        # truncate each w 
        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append( truncation_latent + truncation * (style-truncation_latent)  )
            styles = style_t


        # duplicate and concat into BS * n_latent * code_len 
        if len(styles) == 1:
            latent = styles[0].repeat(1, self.n_latent, 1)  
        elif len(styles) == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)
            latent1 = styles[0].repeat(1, inject_index, 1)
            latent2 = styles[1].repeat(1, self.n_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)
        else:
            latent = torch.cat( styles, 1 )

        return latent


    def __prepare_noise(self, noise, randomize_noise):
        "This is a private function to prepare noise needed during forward"

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [ getattr(self.noises, f'noise_{i}') for i in range(self.num_layers) ]

        return noise



    def forward(self, styles, return_latents=False, inject_index=None, truncation=1, truncation_latent=None, input_type='z', noise=None, randomize_noise=True):

        """
        styles: it should be either list or a tensor.
                List Case:
                    a list containing either z code or w code.
                    and each code (whether z or w, specify by input_type) in this list should be bs*len_of_code.
                    And number of codes should be 1 or 2 or self.n_latent. 
                    When len==1, later this code will be broadcast into bs*self.n_latent*512
                    if it is 2 then it will perform style mixing. If it is self.n_latent, then each of them will 
                    provide style for each layer.
                Tensor Case:
                    then it has to be bs*self.n_latent*code_len, which means it is a w+ code.
                    In this case input_type should be 'w+', and for now we do not support truncate,
                    we assume the input is a ready-to-go latent code from w+ space
            
        return_latents: if true w+ code: bs*self.n_latent*512 tensor will be returned 

        inject_index: int value, it will be specify for style mixing, only will be used when len(styles)==2 

        truncation: whether each w will be truncated 
        
        truncation_latent: if given then it should be calculated from mean_latent function. It has size 1*1*512
                           if truncationm, then this latent must be given 

        input_type: input type of styles, 'z', 'w' 'w+'
        
        noise: if given then recommand to run make_noise first to get noise and then use that as input. if given 
               randomize_noise will be ignored 
         
        randomize_noise: if true then each forward will use different noise, if not a pre-registered fixed noise
                         will be used for each forward.

        """
        if input_type == 'z' or input_type == 'w': 
            assert len(styles) in [1,2,self.n_latent], 'number of styles must be 1, 2 or self.n_latent'
        elif input_type == 'w+':
            assert styles.ndim == 3 and styles.shape[1] == self.n_latent
        else:
            assert False

        latent = self.__prepare_letent(styles, inject_index, truncation, truncation_latent, input_type)
        noise = self.__prepare_noise(noise, randomize_noise)

        # # # start generating # # #  

        out = self.input(latent)
        out = self.conv1(out, latent[:,0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip( self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None







class Discriminator(nn.Module):
    def __init__(self, size, in_c=3, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        log_size = int(math.log(size, 2))
        convs = [ConvLayer(in_c, channels[size], 1)]        

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential( EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
                                           EqualLinear(channels[4], 1) )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view( group, -1, self.stddev_feat, channel // self.stddev_feat, height, width )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

