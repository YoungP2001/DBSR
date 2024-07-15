import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

from torch.nn.modules.utils import _pair, _quadruple


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


# def weights_init_G(m):
#     """ initialize weights of the generator """
#     if m.__class__.__name__.find('Conv') != -1:
#         nn.init.xavier_normal_(m.weight, 0.1)
#         if hasattr(m.bias, 'data'):
#             m.bias.data.fill_(0)

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    # EDSR ???:bn???????????
    """

    def __init__(self, nf=64, res_scale=1.0):
        super(ResidualBlock_noBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True) # ??????????????,0.1 ?????????????????????????,???????????

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out.mul(self.res_scale) # mul?????


class ResidualBlock_BN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nf=64, res_scale=1.0):
        super(ResidualBlock_BN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.bn = nn.BatchNorm2d(nf)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn(self.conv1(x)))
        out = self.conv2(out)
        return identity + out.mul(self.res_scale)


##  Channel Attention (CA) Layer SE?????(Squeeze-and-Excitation Networks)
class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResAttentionBlock(nn.Module):
    def __init__(self, n_feats, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttentionBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, 1, bias=True))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        m.append(CALayer(n_feats, 16))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # # ????
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)        # ?????? [b,c,h,w]==>[b,c,1,1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # 1D??
        y = self.sigmoid(y)
        return x * y.expand_as(x)  # expand_as ?????x??

class Res_ecalock(nn.Module):
    def __init__(self, n_feats, bn=False, act=nn.ReLU(True), res_scale=1,ratio=8):
        super(Res_ecalock, self).__init__()

        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, 1, bias=True))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        m.append(eca_block(n_feats))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


# ???????
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # ??????
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # ??????

        # ??1x1???????
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# ???????
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # ?????????
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # ?????????
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)  # ???????
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)  # ???????

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class ResCBAMBlock(nn.Module):
    def __init__(self, n_feats, bn=False, act=nn.ReLU(True), res_scale=1,ratio=8):
        super(ResCBAMBlock, self).__init__()

        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, 1, bias=True))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        m.append(cbam_block(n_feats, ratio ,kernel_size=3))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


def flow_warp(x, flow, interp_mode="bilinear", padding_mode="zeros"):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output


class Upsampler(nn.Sequential):
    def __init__(self, conv2d, wn, scale, n_feats, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(wn(conv2d(n_feats, 4 * n_feats, 3, bias)))
                m.append(nn.PixelShuffle(2))

                if act == 'relu':
                    m.append(nn.ReLU(inplace=True))

        elif scale == 3:
            m.append(wn(conv2d(n_feats, 9 * n_feats, 3, bias)))
            m.append(nn.PixelShuffle(3))

            if act == 'relu':
                m.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


# ?3D conv
class threeUnit(nn.Module):
    def __init__(self, conv3d, wn, n_feats, bias=True, bn=False, act=nn.ReLU(inplace=True)):
        super(threeUnit, self).__init__()

        self.spatial = wn(conv3d(n_feats, n_feats, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=bias))
        self.spectral = wn(conv3d(n_feats, n_feats, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=bias))

        self.spatial_one = wn(conv3d(n_feats, n_feats, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=bias))
        self.relu = act

    def forward(self, x):
        out = self.spatial(x) + self.spectral(x)
        out = self.relu(out)
        out = self.spatial_one(out)

        return out


class ThreeCNN(nn.Module):
    def __init__(self, n_feats=64):
        super(ThreeCNN, self).__init__()
        self.act = nn.ReLU(inplace=True)
        wn = lambda x: torch.nn.utils.weight_norm(x)
        body_spatial = []
        for i in range(2):
            body_spatial.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))))

        body_spectral = []
        for i in range(2):
            body_spectral.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))))

        self.body_spatial = nn.Sequential(*body_spatial)
        self.body_spectral = nn.Sequential(*body_spectral)

    def forward(self, x):
        out = x
        for i in range(2):

            out = torch.add(self.body_spatial[i](out), self.body_spectral[i](out))
            if i == 0:
                out = self.act(out)

        out = torch.add(out, x)
        return out


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x