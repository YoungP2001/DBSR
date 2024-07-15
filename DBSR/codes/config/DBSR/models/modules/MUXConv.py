
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def _get_padding(kernel_size, stride=1, dilation=1, **_):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def _calc_same_pad(i, k, s, d):
    return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)


def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class ChannelToSpace(nn.Module):

    def __init__(self, upscale_factor=2):
        super().__init__()
        self.bs = upscale_factor

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class SpaceToChannel(nn.Module):

    def __init__(self, downscale_factor=2):
        super().__init__()
        self.bs = downscale_factor

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation,
            groups, bias)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        pad_h = _calc_same_pad(ih, kh, self.stride[0], self.dilation[0])
        pad_w = _calc_same_pad(iw, kw, self.stride[1], self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if _is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = _get_padding(kernel_size, **kwargs)
                return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
            else:
                # dynamic padding
                return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            return nn.Conv2d(in_chs, out_chs, kernel_size, padding=0, **kwargs)
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = _get_padding(kernel_size, **kwargs)
            return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
    else:
        # padding was specified as a number or pair
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


class MuxConv(nn.Module):
    """ MuxConv
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding='', scale_size=[1,1,1,1], groups=1, depthwise=True, **kwargs):
        super(MuxConv, self).__init__()

        scale_size = scale_size if isinstance(scale_size, list) else [scale_size]
        assert len(set(scale_size)) > 1, "use regular convolution for faster inference"

        num_groups = len(scale_size)
        #print('len(scale_size)',len(scale_size))
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size] * num_groups
        groups = groups if isinstance(groups, list) else [groups] * num_groups

        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)

        convs = []
        for k, in_ch, out_ch, scale, _group in zip(kernel_size, in_splits, out_splits, scale_size, groups):
            # print("k, in_ch, out_ch, scale, _group ",k, in_ch, out_ch, scale, _group )
            # padding = (k - 1) // 2
            # print("scale:",scale)
            if scale < 1:  # space-to-channel -> learn -> channel-to-space
                # if depthwise:
                # _group = in_ch * 4
                convs.append(
                    nn.Sequential(
                        SpaceToChannel(2),
                        conv2d_pad(
                            in_ch * 4, out_ch * 4, k, stride=stride,
                            padding=padding, dilation=1, groups=_group, **kwargs),
                        ChannelToSpace(2),
                    )
                )
            elif scale > 1:  # channel-to-space -> learn -> space-to-channel
                # if depthwise:

                convs.append(
                    nn.Sequential(
                        ChannelToSpace(2),
                        conv2d_pad(
                            in_ch // 4, out_ch // 4, k, stride=stride,
                            padding=padding, dilation=1, groups=_group, **kwargs),
                        SpaceToChannel(2),
                    )
                )
            else:
                # if depthwise:

                convs.append(
                    conv2d_pad(
                        in_ch, out_ch, k, stride=stride,
                        padding=padding, dilation=1, groups=_group, **kwargs))
        #print("scale:", scale)

        self.convs = nn.ModuleList(convs)
        self.splits = in_splits
        self.scale_size = scale_size

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        # print('')
        x_out = []
        for spx, conv in zip(x_split, self.convs):
            x_out.append(conv(spx))
        x = torch.cat(x_out, 1)
        return x


"""
temp = torch.randn((16, 64, 32, 32))
group = MuxConv(64, 128,3,1,1,[0.5,0.75,1,2],1)
# print('group=',group)
print(group(temp).size())
print('************************************')
"""
