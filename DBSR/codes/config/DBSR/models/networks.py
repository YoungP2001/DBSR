import logging

import torch
import codes.config.DBSR.models.modules as M
# from models import modules as M

logger = logging.getLogger("base")

# Generator 似乎只有这个用了
def define_G(opt):
    opt_net = opt["network_G"]
    which_model = opt_net["which_model_G"]
    setting = opt_net["setting"]
    netG = getattr(M, which_model)(**setting)  #返回M的属性:which_model的值
    return netG


# Discriminator 似乎没有用
def define_D(opt):
    opt_net = opt["network_D"]
    setting = opt_net["setting"]
    netD = getattr(M, which_model)(**setting)
    return netD


# Perceptual loss 似乎也没有用
def define_F(opt, use_bn=False):
    gpu_ids = opt["gpu_ids"]
    device = torch.device("cuda" if gpu_ids else "cpu")
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = M.VGGFeatureExtractor(
        feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True, device=device
    )
    netF.eval()  # No need to train
    return netF
