import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from models.stylegan2.building_blocks import PixelNorm, EqualLinear, ConstantInput, StyledConv, ConvLayer, ResBlock, ToRGB 
# from models.utils import EdgeDetector, FeatureInterpolator, FeaturePropagator



class Generator(nn.Module):
    def __init__(self, args, device, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01):
        super().__init__()

        self.args = args
        self.device = device

        layers = [PixelNorm()]
        for i in range(args.n_mlp):
            layers.append( EqualLinear( args.style_dim, args.style_dim, lr_mul=lr_mlp, activation='fused_lrelu' )  )
        self.style = nn.Sequential(*layers)


        if self.args.condv=='3' or self.args.condv=='4':
            print('having condition mapping')
            layers = [PixelNorm()]
            for i in range(args.n_mlp):
                if i==0:
                    layers.append( EqualLinear( 10, args.style_dim, lr_mul=lr_mlp, activation='fused_lrelu' )  )
                else:
                    layers.append( EqualLinear( args.style_dim, args.style_dim, lr_mul=lr_mlp, activation='fused_lrelu' )  )
            self.style_c = nn.Sequential(*layers)            

        self.channels = {   4: 512,
                            8: 512,
                            16: 512,
                            32: 512,
                            64: int(256 * args.channel_multiplier),
                            128: int(128 * args.channel_multiplier),
                            256: int(64 * args.channel_multiplier),
                            512: int(32 * args.channel_multiplier),
                            1024: int(16 * args.channel_multiplier) }

        if self.args.condv=='4':
            self.inject_index=10

        final_channel = 6 if args.dataset=='Metal' else 5

        if self.args.no_cond or self.args.condv!='1':
            if not self.args.rand_start:
                self.input = ConstantInput(self.channels[4])
            else:
                print('....................randomize starting layer......................')
        else:
            self.encoder = Encoder(args, device=self.device)

        # if self.args.color_cond:
        #     self.encoder2 = Encoder2(args, device=self.device)

        self.w_over_h = args.scene_size[1] / args.scene_size[0]
        assert self.w_over_h.is_integer(), 'non supported scene_size'
        self.w_over_h = int(self.w_over_h)

        self.log_size = int(math.log(args.scene_size[0], 2)) - int(math.log(self.args.starting_height_size, 2))
        self.num_layers = self.log_size * 2 
        self.n_latent = self.log_size * 2 + 1 

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        expected_out_size = self.args.starting_height_size
        layer_idx = 0 
        for _ in range(self.log_size):
            expected_out_size *= 2
            shape = [1, 1, expected_out_size, expected_out_size*self.w_over_h]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.zeros(*shape))
            self.noises.register_buffer(f'noise_{layer_idx+1}', torch.zeros(*shape))
            layer_idx += 2 

        in_channel = self.channels[self.args.starting_height_size]
        expected_out_size = self.args.starting_height_size     
        in_style_dim = args.style_dim*2 if self.args.condv=='3' else args.style_dim
        for _ in range(self.log_size):  
            expected_out_size *= 2 
            out_channel = self.channels[expected_out_size]
            self.convs.append( StyledConv( in_channel, out_channel, 3, in_style_dim, upsample=True, blur_kernel=blur_kernel, circular=args.circular, circular2=args.circular2 ) )
            # self.convs.append( StyledConv(out_channel, out_channel, 3, args.style_dim, blur_kernel=blur_kernel, circular=args.circular, circular2=args.circular2 ) )
            self.convs.append( StyledConv(out_channel, out_channel, 3, in_style_dim, blur_kernel=blur_kernel, circular=args.circular, circular2=args.circular2 ) )
            # self.to_rgbs.append(ToRGB(out_channel, args.style_dim, out_channel = final_channel, circular=args.circular, circular2=args.circular2))
            self.to_rgbs.append(ToRGB(out_channel, in_style_dim, out_channel = final_channel, circular=args.circular, circular2=args.circular2))
            in_channel = out_channel                               

        
    def make_noise(self):

        expected_out_size = self.args.starting_height_size
        noises = []
        for _ in range(self.log_size):
            expected_out_size *= 2
            noises.append( torch.randn(1, 1, expected_out_size, expected_out_size*self.w_over_h, device=self.device) )
            noises.append( torch.randn(1, 1, expected_out_size, expected_out_size*self.w_over_h, device=self.device) )

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn( n_latent, self.args.style_dim, device=self.device)
        latent = self.style(latent_in).mean(0, keepdim=True).unsqueeze(1)
        return latent

    def get_latent(self, input):
        return self.style(input)

    def __prepare_starting_feature(self, global_pri, styles, input_type, jitter):
        # if self.args.color_cond:
        #     feature, _, _ = self.encoder(global_pri)
        #     z, loss = self.encoder2(color_pri)
        #     if input_type == None:
        #         styles = [z]
        #         input_type = 'z'
        #     return  feature, styles, input_type, loss            
        # else:
        if self.args.no_cond:
            # print(global_pri.shape[0])
            if self.args.rand_start:
                feature = torch.randn(global_pri.shape[0], self.channels[4], 4, 4, device=self.device)
            else:
                feature = self.input(global_pri)

            z = self.args.truncate_z * torch.randn(global_pri.shape[0], self.args.style_dim, device=self.device)
            loss = torch.tensor([0.0], requires_grad=True, device=self.device)
        else:
            if self.args.condv=='2':
                feature = self.input(global_pri)
                input_std, input_m = torch.std_mean(global_pri, dim=(2,3))
                input_cat = torch.cat([input_m, input_std], dim=1)
                z = torch.randn(global_pri.shape[0], self.args.style_dim-10, device=self.device)
                z = torch.cat([z, input_cat], dim=1)
                loss = torch.tensor([0.0], requires_grad=True, device=self.device)
            # c --> map_c --> w_c cat w
            elif self.args.condv=='3' or self.args.condv=='4':
                feature = self.input(global_pri)
                z = self.args.truncate_z * torch.randn(global_pri.shape[0], self.args.style_dim, device=self.device)
                loss = torch.tensor([0.0], requires_grad=True, device=self.device)
            else:
                feature, z, loss = self.encoder(global_pri, jitter=jitter)
        if input_type == None:
            styles = [z]
            input_type = 'z'
        return  feature, styles, input_type, loss

    def __prepare_letent(self, styles, inject_index, truncation, truncation_latent,  input_type, style_c=None):
        "This is a private function to prepare w+ space code needed during forward"
        if input_type == 'z':
            styles = [self.style(s).unsqueeze(1) for s in styles]  # each one is bs*1*512
            # print('style ',styles[0].shape)
            # concat conditional w
            if self.args.condv=='3':
                input_std, input_m = torch.std_mean(style_c, dim=(2,3))
                input_cat = torch.cat([input_m, input_std], dim=1)
                style_c = [self.style_c(input_cat).unsqueeze(1)]  # each one is bs*1*512
                styles = [torch.cat([s,c], dim=-1) for s,c in zip(styles,style_c)]

            elif self.args.condv=='4':
                input_std, input_m = torch.std_mean(style_c, dim=(2,3))
                input_cat = torch.cat([input_m, input_std], dim=1)
                style_c = [self.style_c(input_cat).unsqueeze(1)]  # each one is bs*1*512

            # print('style ',styles[0].shape)

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
            if self.args.condv=='4':
                latent = styles[0].repeat(1, self.n_latent - self.inject_index, 1) 
                latent_c = style_c[0].repeat(1, self.inject_index, 1) 
                print('mixing cond style')
                latent = torch.cat([latent, latent_c], 1)
            else:
                latent = styles[0].repeat(1, self.n_latent, 1) 

        elif len(styles) == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)
            latent1 = styles[0].repeat(1, inject_index, 1)
            latent2 = styles[1].repeat(1, self.n_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)
        else:
            latent = torch.cat( styles, 1 )
        # print('latent ',latent.shape)

        return latent

    def __prepare_noise(self, noise, randomize_noise):
        "This is a private function to prepare noise needed during forward"

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [ getattr(self.noises, f'noise_{i}') for i in range(self.num_layers) ]

        return noise
  
    def forward(self, global_pri, styles=None, return_latents=False, inject_index=None, truncation=1, truncation_latent=None, input_type=None, noise=None, randomize_noise=True, return_loss=True, shiftN=None, jitter=None, out_inter=False):

        """
        global_pri: a tensor with the shape BS*C*self.prior_size*self.prior_size. Here, in background training,
                    it should be semantic map + edge map, so it should have channel 151+1 

        styles: it should be either list or a tensor or None.
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
                None Case:
                    Then z code will be derived from global_pri also. In this case input_type shuold be None
            
        return_latents: if true w+ code: bs*self.n_latent*512 tensor, will be returned 

        inject_index: int value, it will be specify for style mixing, only will be used when len(styles)==2 

        truncation: whether each w will be truncated 
        
        truncation_latent: if given then it should be calculated from mean_latent function. It has size 1*1*512
                           if truncation, then this latent must be given 

        input_type: input type of styles, None, 'z', 'w' 'w+'
        
        noise: if given then recommand to run make_noise first to get noise and then use that as input. if given 
               randomize_noise will be ignored 
         
        randomize_noise: if true then each forward will use different noise, if not a pre-registered fixed noise
                         will be used for each forward.

        return_loss: if return kl loss.  

        """
        if input_type == 'z' or input_type == 'w': 
            assert len(styles) in [1,2,self.n_latent], f'number of styles must be 1, 2 or self.n_latent but got {len(styles)}'
        elif input_type == 'w+':
            # print('styles.ndim: ', styles.ndim, ' styles.shape[1]: ', styles.shape[1], ' self.n_latent: ', self.n_latent)
            assert styles.ndim == 3 and styles.shape[1] == self.n_latent
        elif input_type == None:
            assert styles == None
        else:
            assert False, 'not supported input_type'

        start_feature, styles, input_type, loss = self.__prepare_starting_feature(global_pri, styles, input_type, jitter)
        # print( 'starting feature: ',start_feature[0,0,0])
        latent = self.__prepare_letent(styles, inject_index, truncation, truncation_latent, input_type, style_c=global_pri if self.args.condv=='3' or self.args.condv=='4' else None)
        noise = self.__prepare_noise(noise, randomize_noise)
        
        # if noise[0] is not None:
        #     print('noise ', noise[0][0,0,5,5])
        # else:
        #     print('noise is none')
        # # # start generating # # #  

        out = start_feature
        skip = None

        assert out.isnan().any()==False, 'start_feature nan'

        inter_feature = []
        i = 0
        for conv1, conv2, noise1, noise2, to_rgb in zip( self.convs[::2], self.convs[1::2], noise[::2], noise[1::2], self.to_rgbs ):
            # print('out: ',out.shape)
            out = conv1(out, latent[:, i], noise=noise1, shiftN=shiftN)  
            assert out.isnan().any()==False, 'out1 nan'

            # print('out: ',out[0,0, 20, 20:25])
            out = conv2(out, latent[:, i + 1], noise=noise2, shiftN=shiftN)   
            # print('out: ',out[0,0, 20, 20:25])
            assert out.isnan().any()==False, 'out2 nan'

            skip = to_rgb(out, latent[:, i + 2], skip)
            # print('skip: ',skip[0, 0, 20, 20:25])
            assert skip.isnan().any()==False, 'skip nan'

            if out_inter:
                inter_feature.append(out)

            i += 2

        image = F.tanh(skip)

        output = { 'image': image }
        if return_latents:
            output['latent'] =  latent  
        if return_loss:
            output['klloss'] =  loss 


        if out_inter:
            return output, inter_feature
        else:
            return output





# class Padder():
#     def __init__(self, scene_size):

#         if scene_size[0]<scene_size[1]:
#             # height is smaller, so pad up and bottem 
#             p = int( (scene_size[1]-scene_size[0])/2 )
#             self.pad = nn.ZeroPad2d((0, 0, p, p))
#         else:
#             # width is smaller, so pad left and right
#             p = int( (scene_size[0]-scene_size[1])/2 )
#             self.pad = nn.ZeroPad2d((p, p, 0, 0))

#     def __call__(self, x):
        
#         return self.pad(x)
            
    
class Padder():
    def __init__(self, scene_size):

        print('scene size will be splitted in D, thus input and output batch dimension is not the same')

        if scene_size[0]<scene_size[1]:
            # height is smaller, so split width: 3rd dimension 
            self.dim = 3 
            self.size = scene_size[0]
        else:
            # width is smaller, so split height: 2nd dimension
            self.dim = 2 
            self.size = scene_size[1]

    def __call__(self, x):
        xx = torch.split(x, self.size, self.dim)
        return torch.cat( [xx[0], xx[1]], dim=0 )
            
class Discriminator(nn.Module):
    def __init__(self, args, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.args=args
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * args.channel_multiplier,
            128: 128 * args.channel_multiplier,
            256: 64 * args.channel_multiplier,
            512: 32 * args.channel_multiplier,
            1024: 16 * args.channel_multiplier,
        }

        # if args.scalar_cond and args.cond_D:
        #     mlp = []

        #     mlp.append(EqualLinear(10, 10, lr_mul=0.01, activation='fused_lrelu'))
        #     mlp.append(EqualLinear(10, 10, lr_mul=0.01, activation='fused_lrelu'))
        #     mlp.append(EqualLinear(10, 10, lr_mul=0.01, activation='fused_lrelu'))
        #     mlp.append(EqualLinear(10, 10, lr_mul=0.01, activation='fused_lrelu'))
        #     # mlp.append(EqualLinear(10, 10, activation='fused_lrelu'))
        #     # mlp.append(EqualLinear(10, 10, activation='fused_lrelu'))
        #     # mlp.append(EqualLinear(10, 10, activation='fused_lrelu'))
        #     # mlp.append(EqualLinear(512, 512*512, lr_mul=0.01, activation='fused_lrelu') )
        #     self.mlp = nn.Sequential(*mlp)


        if args.scene_size[0] == args.scene_size[1]:
            input_size = args.scene_size[0]
            self.need_handle_size = False
        else:
            input_size = min(args.scene_size)  ########## if padding, then this should be max
            self.need_handle_size = True
            self.padder = Padder(args.scene_size)


        log_size = int( math.log(input_size,2) )

        in_c = 6 if args.dataset=='Metal' else 5

        if args.cond_D:
            if args.color_cond:
                in_c += 3
            else: 
                if args.scalar_cond:
                    in_c += 10
                else:
                    in_c += 1

        convs = [ ConvLayer(in_c, channels[input_size], 1) ]        

        in_channel = channels[input_size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel, circular = self.args.circular, circular2 = self.args.circular2, dk_size=self.args.dk_size))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, circular = self.args.circular, circular2 = self.args.circular2)
        self.final_linear = nn.Sequential( EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
                                           EqualLinear(channels[4], 1) )

    def forward(self, input):

        if self.need_handle_size:
            input = self.padder(input)

        # if self.args.scalar_cond and self.args.cond_D:
        #     # compute the statistics of input
        #     input1 = input[:,:5,:,:]

        #     input_std, input_m = torch.std_mean(input[:,-5:,:,:], dim=(2,3))
        #     # input_cat = torch.cat([input_m, input_m], dim=1)
        #     input_cat=input_m.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, input.shape[-2],input.shape[-1])
        #     input = torch.cat([input1, input_cat], dim=1)
        #     # print(input.shape)

        #     assert input1.isnan().any()==False, 'input1 nan'
        #     assert input_cat.isnan().any()==False, 'input_cat nan'

        # print('input', input.shape)
        out = self.convs(input)
        # print('out', out.shape)
        assert out.isnan().any()==False, 'nan convs'

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view( group, -1, self.stddev_feat, channel // self.stddev_feat, height, width )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        assert out.isnan().any()==False, 'nan cat'

        out = self.final_conv(out)
        assert out.isnan().any()==False, 'nan final_conv'

        out = out.view(batch, -1)
        out = self.final_linear(out)
        assert out.isnan().any()==False, 'nan final_linear'

        return out

class SmallUnet(nn.Module):
    "Here we aggregate feature from different resolution. It is actually an up branch of Unet"
    def __init__(self, size_to_channel, circular=False, circular2=False):
        super().__init__()
        
        self.convs = nn.ModuleList()
        
        size = 4 
        extra_channel = 0 
        
        for i in range(  len(size_to_channel)  ):
            in_channel = size_to_channel[size] + extra_channel
            upsample = i != (len(size_to_channel)-1) 
            self.convs.append(     ConvLayer(in_channel, 512, 3, upsample=upsample, circular=circular, circular2=circular2)      )
            size *= 2
            extra_channel = 512
    
    def forward(self, feature_list):
        "feature_list should be ordered from small to big: BS*C1*4*?, BS*C2*8*?, BS*C3*16*?,..."
     
        for conv, feature in zip(self.convs, feature_list):
            if feature.shape[2] != 4:  
                # print('feature2: ', feature.shape)
                # print('previos2: ', previos.shape)
                feature = torch.cat( [feature,previos], dim=1 )
            # print('feature1: ', feature.shape)
            previos = conv(feature)
            # print('previos1: ', previos.shape)
        return previos


class Encoder(nn.Module):
    def __init__(self, args, device ,blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.args = args

        self.device = device
        channels = { 4: 512,
                     8: 512,
                     16: 512,
                     32: 512,
                     64: 512,
                     128: int(128 * args.channel_multiplier),
                     256: int(64 * args.channel_multiplier),
                     512: int(32 * args.channel_multiplier),
                     1024: int(16 * args.channel_multiplier) }

        # self.convs1 = ConvLayer(args.number_of_semantic+1, channels[args.scene_size[0]], 1)  # this 1 is edge map
        in_c = 3 if args.color_cond else 1

        # if use scalar as condition, 
        if args.scalar_cond:

            layers = []

            if args.starting_height_size==4:
                mlp_channels = [10, # 1
                                128, # 3
                                512, # 3
                                512, # 3
                                512, # 3
                                512*4*4, # 3
                                ] 
                n_layer=5
            elif args.starting_height_size==32:
                mlp_channels = [10, # 1
                                128, # 2
                                512, # 3
                                512, # 4
                                512, # 5
                                512*32*32, # 8
                                ] 
                n_layer=5                

            for i in range(n_layer):
                layers.append( EqualLinear( mlp_channels[i], mlp_channels[i+1], lr_mul=0.01, activation='fused_lrelu' )  )
            self.mlp = nn.Sequential(*layers)

        else:
            print('.......use encoder ................')
            self.convs1 = ConvLayer(in_c, channels[args.scene_size[0]], 1)  

            log_size = int(math.log(args.scene_size[0], 2))

            in_channel = channels[args.scene_size[0]]
            size_to_channel = {} # indicate from which resolution we provide spatial feature 
            self.convs2 = nn.ModuleList()
            for i in range(log_size, 2, -1):
                out_size = 2 ** (i-1)
                out_channel = channels[ out_size ]
                if 4 <= out_size <= args.starting_height_size: 
                    size_to_channel[out_size] = out_channel 
                self.convs2.append(ResBlock(in_channel, out_channel, blur_kernel, circular=self.args.circular, circular2=self.args.circular2, dk_size=self.args.dk_size))
                in_channel = out_channel

            self.unet = SmallUnet( size_to_channel, circular=self.args.circular, circular2=self.args.circular2 )

            w_over_h = args.scene_size[1] / args.scene_size[0]
            assert w_over_h.is_integer(), 'non supported scene_size'

            if not args.nocond_z:
                if args.extract_model:
                    self.final_linear = EqualLinear(channels[4] * 2 * 2 * int(w_over_h), channels[4], activation='fused_lrelu')
                else:
                    self.final_linear = EqualLinear(channels[4] * 4 * 4 * int(w_over_h), channels[4], activation='fused_lrelu')
                self.mu_linear = EqualLinear(channels[4], args.style_dim)
                self.var_linear = EqualLinear(channels[4], args.style_dim)

        # mapping scalar to condition
        # def scalar_cond(self):
        #     self.mlp.append(EqualLinear(10, 512, activation='fused_lrelu') )
            # self.mlp.append(EqualLinear(512, 512, activation='fused_lrelu') )
            # linear1 = EqualLinear(512, 512*64, activation='fused_lrelu')
            # linear1 = EqualLinear(512*64, 512*256, activation='fused_lrelu')
            # linear1 = EqualLinear(512*256, 512*512, activation='fused_lrelu')


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)         
        return eps.mul(std) + mu
    

    def get_kl_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


    def forward(self, input, jitter=None):
        batch = input.shape[0]
        intermediate_feature = []

        # if use scalar as condition
        if self.args.scalar_cond:

            # compute the statistics of input
            input_std, input_m = torch.std_mean(input, dim=(2,3))
            input_cat = torch.cat([input_m, input_std], dim=1)

            # add jittering if needed
            if jitter is not None:
                # print('jitter')
                # alpha = (jitter==0).long()
                # input_cat = input_cat*alpha + (1-alpha)*(input_cat + jitter)*0.5 
                input_cat = (input_cat + jitter)*0.5 
                # strong jitter

            out = self.mlp(input_cat)
            # assert out.isnan().any()==False, 'out nan encoder'

            starting_feature=out.view(batch, 512, self.args.starting_height_size, self.args.starting_height_size)
            # print('featuer shape ',starting_feature.shape)

            z = self.args.truncate_z * torch.randn(out.shape[0], self.args.style_dim, device=self.device)
            # loss = self.get_kl_loss(z, z)*0.0 # no 
            loss = torch.tensor([0.0], requires_grad=True, device=self.device)

        else:

            out = self.convs1(input)

            # print('out: ',out.shape)
            for conv in self.convs2:
                out = conv(out)
                # print('out: ',out.shape)
                if 4 <= out.shape[2] <= self.args.starting_height_size:
                    intermediate_feature.append( out ) 
            starting_feature = self.unet( intermediate_feature[::-1] )


            # if condition z
            if not self.args.nocond_z:
                out = self.final_linear( out.view(batch, -1))
                mu = self.mu_linear(out)
                logvar = self.var_linear(out)

                z = self.reparameterize(mu, logvar)
                loss = self.get_kl_loss(mu, logvar)
            # if no condition z
            else:
                # print('out: ',out.shape[0], self.args.style_dim)
                z = self.args.truncate_z * torch.randn(out.shape[0], self.args.style_dim, device=self.device)
                # loss = self.get_kl_loss(z, z)*0.0 # no 
                loss = torch.tensor([0.0], requires_grad=True, device=self.device)
                # print('z ',z.shape)

        return starting_feature, z, loss

