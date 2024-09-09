import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from .module_util import *
import math
from .utils_deblur_winear import get_uperleft_denominator
from .MUXConv import MuxConv

## fusion feature and winear filter

def _to_4d_tensor(x, depth_stride=None):
    """Converts a 5d tensor to 4d by stacking
    the batch and depth dimensions.
    NxDxCxHxW  """
    # x = x.transpose(0, 2)  # swap batch and depth dimensions: NxDxCxHxW
    if depth_stride:
        x = x[::depth_stride]  # downsample feature maps along depth dimension,???,depth_stride???
    depth = x.size()[1]  # D
    # x = x.permute(0, 1, 0, 3, 4)  # NxDxCxHxW
    x = torch.split(x, 1, dim=0)  # split along batch dimension: NxDxCxHxW=> N*[1xDxCxHxW]
    x = torch.cat(x, 1)  # concatenate along depth dimension: N*[1xDxCxHxW] => 1x(N*D)xCxHxW
    x = x.squeeze(0)  # 1x(N*D)xCxHxW => (N*D)xCxHxW
    return x, depth


def _to_5d_tensor(x, depth):
    """Converts a 4d tensor back to 5d by splitting
    the batch dimension to restore the depth dimension."""
    x = torch.split(x, depth)  # (N*D)xCxHxW => N*[DxCxHxW]
    x = torch.stack(x, dim=0)  # re-instate the batch dimension: NxDxCxHxW
    # x = x.transpose(1, 2)  # swap back depth and channel dimensions: NxDxCxHxW => NxCxDxHxW
    return x


class SSB(nn.Module):
    def __init__(self, nf, res_scale):  # , conv=default_conv
        super(SSB, self).__init__()
        self.spa = ResidualBlock_noBN(nf, res_scale=res_scale)
        self.spc = ResAttentionBlock(nf, res_scale=res_scale)

    def forward(self, x):
        # return self.spc(x)+self.spa(x)
        return self.spc(self.spa(x))


class SSPN(nn.Module):
    def __init__(self, n_feats, n_blocks, res_scale):
        super(SSPN, self).__init__()

        m = []

        for i in range(n_blocks):
            m.append(SSB(n_feats, res_scale=res_scale))

        self.net = nn.Sequential(*m)

    def forward(self, x):
        res = self.net(x)
        res += x

        return res


class Spa_Spe_Unit(nn.Module):
    def __init__(self, n_feats=64, channel_feats=128, res_scale=1, in_nc=31):
        super(Spa_Spe_Unit, self).__init__()
        # self.act = nn.ReLU(inplace=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        wn = lambda x: torch.nn.utils.weight_norm(x)
        kernel_size = 3
        self.body_spatial = nn.Conv3d(n_feats, n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.body_spectral = nn.Conv3d(n_feats, n_feats, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))

        self.tail = SSB(channel_feats, res_scale)

    def forward(self, x):
        # x = x.unsqueeze(1)  # 4d->5d, Bx1xCxHxW
        out = x

        spa = self.body_spatial(out)
        spe = self.body_spectral(out)
        out = torch.add(spa, spe)
        out = self.act(out)


        # out = self.spe_last(out)
        out = torch.add(out, x)
        # print("out2",out.shape)
        out, depth = _to_4d_tensor(out, depth_stride=1)
        out = self.tail(out)
        out = _to_5d_tensor(out, depth)
        # print(out.shape)

        return out


# a single branch of proposed SSPSR
class Final_Block(nn.Module):
    def __init__(self, in_nc, n_feats, n_blocks, res_scale, channel_feats):
        super(Final_Block, self).__init__()
        kernel_size = 3
        n = []
        channel_feats = 128
        self.head = nn.Conv2d(in_nc, channel_feats, 3, 1, 1)
        self.increase_D = nn.Conv3d(1, n_feats, kernel_size=(1, 1, 1), stride=1)
        for i in range(n_blocks):
            n.append(Spa_Spe_Unit(n_feats, channel_feats, res_scale, in_nc))
        self.body = nn.Sequential(*n)
        self.reduce_D = nn.Conv3d(n_feats, 1, kernel_size=(1, 1, 1), stride=1)

    def forward(self, x):
        x = self.head(x)
        y = x.unsqueeze(1)  # 4d->5d, Bx1xCxHxW
        y = self.increase_D(y)
        # print("y",y.shape)
        y = self.body(y)
        y = self.reduce_D(y)
        y = y.squeeze(1) + x
        return y



class DWDN(nn.Module):
    def __init__(self, nf, reduction=4):
        super().__init__()

        self.reduce_feature = nn.Conv2d(nf, nf // reduction, 1, 1, 0)  # ???????4?
        self.expand_feature = nn.Conv2d(nf // reduction, nf, 1, 1, 0)

    def forward(self, x, kernel):

        w_feats=self.reduce_feature(x)
        clear_features = torch.zeros(w_feats.size()).to(x.device)
        ks = kernel.shape[-1]
        dim = (ks, ks, ks, ks)
        feature_pad = F.pad(w_feats, dim, "replicate")
        #
        for i in range(feature_pad.shape[1]):
            blur_feature_ch = feature_pad[:, i:i + 1, :, :]
            clear_feature_ch = get_uperleft_denominator(blur_feature_ch, kernel)
            clear_features[:, i:i + 1, :, :] = clear_feature_ch[:, :, ks:-ks, ks:-ks]  # ks:-ks=21:-21,

        x = self.expand_feature(clear_features)
        return x


class DPCAB(nn.Module):
    def __init__(self, nf1, nf2, ksize1=3, ksize2=3, reduction=4):
        super().__init__()

        basic_block1 = functools.partial(ResidualBlock_noBN, nf=nf1)
        self.body1 = make_layer(basic_block1, 2)  # 3???

        basic_block2 = functools.partial(ResidualBlock_noBN, nf=nf2)
        self.body2 = make_layer(basic_block2, 2)  # 3???

        self.CA_head1 = nn.Conv2d(nf1 + nf2, nf1, ksize1, 1, ksize1 // 2)
        self.CA_head2 = nn.Conv2d(2*nf1, nf1, ksize1, 1, ksize1 // 2)

        self.CA_body1 = CALayer(nf1, reduction)
        self.CA_body2 = CALayer(nf2, reduction)

    def forward(self, x, x_last=[]):
        f1 = self.body1(x[0])
        f2 = self.body2(x[1])

        ca_head_f1 = self.CA_head1(torch.cat([f1, f2], dim=1))
        # ??????????
        if len(x_last)!=0:
            ca_head_f1 = self.CA_head2(torch.cat([ca_head_f1, x_last], dim=1))

        ca_f1 = self.CA_body1(ca_head_f1)
        ca_f2 = self.CA_body2(f2)

        x[0] = x[0] + ca_f1
        x[1] = x[1] + ca_f2
        return x


class DPCAG(nn.Module):
    def __init__(self, nf1, nf2, ksize1, ksize2, nb):
        super().__init__()
        self.nb = nb

        self.body = DPCAB(nf1, nf2, ksize1, ksize2)

    def forward(self, x,f_last):
        y = x
        y_last = []

        for i in range(self.nb):
            if type(f_last) ==list:
                y = self.body(y, f_last[i])  # y=x: inputs = [f1_local, hr_bic,f2]
            else:
                y = self.body(y)
            y_last.append(y[0])

        y[0] = x[0] + y[0]
        y[1] = x[1] + y[1]

        return y, y_last



class BranchUnit(nn.Module):
    def __init__(
            self, in_nc=1, nf=64, nb=8, ng=1, reduction=4
    ):
        super(BranchUnit, self).__init__()
        self.num_blocks = nb

        nf2 = nf // reduction

        self.conv_first = nn.Conv2d(in_nc, nf, 3, stride=1, padding=1)
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.feature_block = make_layer(basic_block, 3)  # 3???

        self.head1 = nn.Conv2d(nf, nf2, 3, 1, 1)  # CR,?????,??,?Gy,???16=64/4
        self.head2 = DWDN(nf, reduction=reduction)  #

        self.body = DPCAG(nf, nf2, 3, 3, nb)

    def forward(self, input, kernel,f_last):
        # B, C, H, W = input.size()  # I_LR batch

        f = self.conv_first(input)  # ?????,?????batch?????,??????????
        feature = self.feature_block(f)
        f1 = self.head1(feature)  # ??,CR
        f2 = self.head2(feature, kernel)  

        inputs = [f2, f1]
        f12,f_last = self.body(inputs,f_last)

        return f, f12, f_last
        # return torch.clamp(f, min=self.min, max=self.max)


class Restorer(nn.Module):
    def __init__(
            self, in_nc=1, nf=64, nb=8, ng=1, scale=4, input_para=10, reduction=4, min=0.0, max=1.0, use_share=True,
            n_subs=3, n_ovls=1, n_SSB_blocks=3, res_scale=0.1, final_feats=64, channel_feats=128):
        super(Restorer, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb

        out_nc = in_nc  # ???????64
        nf2 = nf // reduction
        self.shared = use_share

        ## Divide Groups
        # # calculate the group number (the number of branch networks)
        self.Group = math.ceil((in_nc - n_ovls) / (n_subs - n_ovls))
        # calculate group indices
        self.start_idx = []
        self.end_idx = []

        for g in range(self.Group):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > in_nc:
                end_ind = in_nc
                sta_ind = in_nc - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        if self.shared:
            self.branch = BranchUnit(n_subs, nf, nb, ng, reduction=4)
            # up_scale=n_scale//2 means that we upsample the LR input n_scale//2 at the branch network, and then conduct 2 times upsampleing at the global network
        else:
            self.branch = nn.ModuleList()
            for i in range(self.G):
                self.branch.append(BranchUnit(n_subs, nf, nb, ng, reduction=4))

        self.fusion1 = nn.Conv2d(nf + nf2, nf, 3, 1, 1)
        self.fusion2 = nn.Conv2d(nf, n_subs, 3, 1, 1)

        self.final_Block = Final_Block(in_nc, n_feats=final_feats, n_blocks=n_SSB_blocks, res_scale=res_scale,
                                       channel_feats=channel_feats)
        self.longres = nn.Conv2d(in_nc, final_feats, 3, 1, 1)
        # self.final = nn.Conv2d(final_feats, in_nc, 3, 1, 1)
        # self.fusion = CCALayer(nf, nf, reduction)

        # channels=final_feats
        channels = channel_feats
        s = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                s.append(nn.Conv2d(channels, 4 * channels, 3, 1, 1, bias=True))
                s.append(nn.PixelShuffle(2))
            s.append(nn.Conv2d(channels, out_nc, 3, 1, 1))
            self.upscale = nn.Sequential(*s)


        elif scale == 1:
            self.upscale = nn.Conv2d(channels, out_nc, 3, 1, 1)

        else:  # x2, x3
            self.upscale = nn.Sequential(
                nn.Conv2d(channels, channels * scale ** 2, 3, 1, 1, bias=True),
                nn.PixelShuffle(scale),
                nn.Conv2d(channels, out_nc, 3, 1, 1),
            )

    def forward(self, input, kernel):
        b, c, h, w = input.shape

        y = torch.zeros(b, c, h, w).cuda()

        channel_counter = torch.zeros(c).cuda()
        f_last = [None] * (self.Group + 1)
        for g in range(self.Group):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]
            xi = input[:, sta_ind:end_ind, :, :]

            if self.shared:
                f, f12 ,f_last[g+1]= self.branch(xi, kernel, f_last[g])

            else:
                f, f12 ,f_last[g+1]= self.branch[g](xi, kernel,f_last[g])
            inputs = []
            f2, f1 = f12
            f = self.fusion1(torch.cat([f1, f2], dim=1)) + f
            f = self.fusion2(f)


            y[:, sta_ind:end_ind, :, :] += f
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1

        y = y / channel_counter.unsqueeze(1).unsqueeze(2)  # y????{b,c,h,w}

        y = self.final_Block(y)
        # y1 = self.longres(input)
        # y = y + y1
        out = self.upscale(y)

        return torch.clamp(out, min=self.min, max=self.max)


class ResMcblock(nn.Module):
    def __init__(
            self, nf=64, kernel_size=3, scale_size=[0.5, 1, 1, 2], groups=[1, 1, 1, 1]
    ):
        super(ResMcblock, self).__init__()

        self.mconv1=MuxConv(nf, nf,
                kernel_size=kernel_size, stride=1, padding='same', scale_size=scale_size,groups=groups, depthwise=True)
        self.conv1d1 = nn.Conv2d(nf, nf, 1, bias=True)
        self.act=nn.LeakyReLU(0.1, inplace=True)
        self.mconv2=MuxConv(nf, nf,
                kernel_size=kernel_size, stride=1, padding='same', scale_size=scale_size,groups=groups, depthwise=True)
        self.conv1d2 = nn.Conv2d(nf, nf, 1, bias=True)

    def forward(self, x):
        identity = x
        x=self.mconv1(x)
        x=self.conv1d1(x)
        x=self.act(x)
        x=self.mconv2(x)
        x=self.conv1d2(x)

        return identity+x.mul(0.1)



class Estimator(nn.Module):
    def __init__(
            self, in_nc=1, nf=64, para_len=256, num_blocks=3, kernel_size=4, filter_structures=[]
    ):
        super(Estimator, self).__init__()

        self.filter_structures = filter_structures
        self.ksize = kernel_size
        self.G_chan = 16
        self.in_nc = in_nc
        #basic_block = functools.partial(ResidualBlock_noBN, nf=nf)  # ??ResidualBlock_noBN?????????,??nf???????????????
        basic_block = functools.partial(ResMcblock, nf=nf)

        self.head = nn.Sequential(
            nn.Conv2d(in_nc, nf, 7, 1, 3)
        )
        ## ???3?
        self.body= nn.Sequential(
            make_layer(basic_block, num_blocks)
        )


        self.tail = nn.Sequential(
            nn.Conv2d(nf, nf, 3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, 3),
            nn.AdaptiveAvgPool2d((1, 1)),  # B*nf*1*1
            nn.Conv2d(nf, para_len, 1),  # B*para_len*1*1
            nn.Flatten(),  # nn.Flatten()??1?????????,?????,flat B*para_len*1*1 to 1d = B*para_len
        )

        # ????,???????kernels
        self.dec = nn.ModuleList()
        for i, f_size in enumerate(self.filter_structures):
            if i == 0:
                in_chan = in_nc
            elif i == len(self.filter_structures) - 1:
                in_chan = in_nc
            else:
                in_chan = self.G_chan
            # from 1d:b*para_len to G_chan * in_chan * f_size^2, and then reshape to B*C*kernel_sine*kernel_size
            self.dec.append(nn.Linear(para_len, self.G_chan * in_chan * f_size ** 2))  # fully connected layers,FC

        self.apply(initialize_weights)

    def calc_curr_k(self, kernels, batch):
        """given a generator network, the function calculates the kernel it is imitating"""
        delta = torch.ones([1, batch * self.in_nc]).unsqueeze(-1).unsqueeze(
            -1).cuda()  # ?1??,?????[1, batch*self.in_nc, 1, 1]
        # the way of getting a identity kernel,because for identity kernel only 1 in the center, so unsqueeze(-1) twice can get this(only 1 in the center, other places are 0)

        # ?????????4????
        for ind, w in enumerate(kernels):
            # if kernel is the first, it will convolve with a delta kernel (identity kernel ???)
            curr_k = F.conv2d(delta, w, padding=self.ksize - 1, groups=batch) if ind == 0 else F.conv2d(curr_k, w,
                                                                                                        groups=batch)
        curr_k = curr_k.reshape(batch, self.in_nc, self.ksize, self.ksize).flip(
            [2, 3])  # flip 2d and 3d, maybe calculating Convolution needs kernel to reverse
        return curr_k  # ????

    def forward(self, LR):
        batch, channel = LR.shape[0:2]
        f1 = self.head(LR)
        f = self.body(f1) + f1  # 3????????????????f1

        latent_kernel = self.tail(f)

        kernels = [self.dec[0](latent_kernel).reshape(
            batch * self.G_chan,
            channel,
            self.filter_structures[0],
            self.filter_structures[0])]

        for i in range(1, len(self.filter_structures) - 1):
            kernels.append(self.dec[i](latent_kernel).reshape(
                batch * self.G_chan,
                self.G_chan,
                self.filter_structures[i],
                self.filter_structures[i]))

        # so kernels is a list
        kernels.append(self.dec[-1](latent_kernel).reshape(
            batch * channel,
            self.G_chan,
            self.filter_structures[-1],
            self.filter_structures[-1]))

        K = self.calc_curr_k(kernels, batch).mean(dim=1, keepdim=True)  # ?channel????

        # for anisox2
        # K = F.softmax(K.flatten(start_dim=1), dim=1)
        # K = K.view(batch, 1, self.ksize, self.ksize)

        K = K / torch.sum(K, dim=(2, 3), keepdim=True)  # ?K??2???3??????????,??????1?

        return K


# ??init????????????,forward???????????????
class DBSR(nn.Module):
    def __init__(
            self,
            nf=64,
            nb=16,
            ng=5,
            in_nc=31,  # ????????,?3???31
            # n_colors=31, # number of bands
            n_subs=3,  # the sub band in each group
            n_ovls=1,  # the number of overlap
            n_SSB_blocks=2,
            res_scale=0.1,
            final_feats=64,
            channel_feats=128,
            reduction=4,
            upscale=4,
            input_para=128,
            kernel_size=21,
            pca_matrix_path=None,
            ker_ex_numblock=3,
    ):
        super(DBSR, self).__init__()

        self.ksize = kernel_size
        self.scale = upscale

        if kernel_size == 21:
            filter_structures = [11, 7, 5, 1]  # for iso kernels all
        elif kernel_size == 11:
            filter_structures = [7, 3, 3, 1]  # for aniso kernels x2
        elif kernel_size == 31:
            filter_structures = [11, 9, 7, 5, 3]  # for aniso kernels x4
        elif kernel_size == 35:
            filter_structures = [11, 9, 7, 7, 5]  # for aniso kernels x8
        else:
            print("Please check your kernel size, or reset a group filters for DDLK")

        self.Restorer = Restorer(
            nf=nf, in_nc=in_nc, nb=nb, ng=ng, scale=self.scale, input_para=input_para, reduction=reduction,
            n_subs=n_subs,
            n_ovls=n_ovls, n_SSB_blocks=n_SSB_blocks, res_scale=res_scale, final_feats=final_feats
        )
        self.Estimator = Estimator(
            kernel_size=kernel_size, para_len=input_para*2, in_nc=in_nc, nf=nf*4, filter_structures=filter_structures,num_blocks=ker_ex_numblock
        )

    def forward(self, lr): 

        kernel = self.Estimator(lr)  
        sr = self.Restorer(lr, kernel.detach())  
        return sr, kernel
