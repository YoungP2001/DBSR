import os
import sys
import cv2
import numpy as np
import torch
import glob
import random
import scipy.io
from scipy.io import loadmat
from PIL import Image
try:
    # sys.path.append('..')
    sys.path.append('/home/lwd/code_yp/DBSR-SR')
    from codes.data_util import imresize
    import codes.utils as util
except ImportError:
    from codes.data.data_util import imresize
    import codes.utils as util

    pass


def generate_mod_LR_bic():
    # set parameters
    up_scale = 8
    #mod_scale = 4
    # set data dir

    # savedir = "/data/yp/HSI/Data/Harvard/Train/4_blur/"
    # sourcedir = "/data/yp/HSI/Data/Harvard/Train/4/"
    savedir = "/home/lwd/Dataset_yp/HSISR/Chikusei/8/Chikusei_x8_blur/trains_large"
    sourcedir = "/home/lwd/Dataset_yp/HSISR/Chikusei/8/Chikusei_x8/trains_large/"

    if not os.path.exists(savedir):
        os.makedirs(savedir)
        print("Folder created")
    else:
        print("Folder already exists")

    i = 0
    # load PCA matrix of enough kernel
    print("load PCA matrix")
    pca_matrix = torch.load(
        "../../pca_matrix/DBSR/pca_matrix.pth", map_location=lambda storage, loc: storage
    )
    print("PCA matrix shape: {}".format(pca_matrix.shape))

    degradation_setting = {
        "random_kernel": True,
        "code_length": 10,
        "ksize": 21,
        "pca_matrix": pca_matrix,
        "scale": up_scale,
        "cuda": True,
        "rate_iso": 1.0,
        "random_disturb": False,
        "sig_min": 0.2,
        "sig_max": 4.0
    }

    # pca_matrix = torch.load(
    #     "../../pca_matrix/DBSR/pca_aniso_matrix_x4.pth", map_location=lambda storage, loc: storage
    # )
    #
    # print("PCA matrix shape: {}".format(pca_matrix.shape))
    # degradation_setting = {
    #     "random_kernel": True,
    #     "code_length": 10,
    #     "ksize": 31,
    #     "pca_matrix": pca_matrix,
    #     "scale": up_scale,
    #     "cuda": True,
    #     "rate_iso": 0,
    #     "random_disturb": True,
    #     "sig_min": 0.6,
    #     "sig_max": 5
    # }

    files = os.listdir(sourcedir)
    # num_files_to_pick = 100
    # selected_files = random.sample(files, num_files_to_pick)
    for file in files:
        print("Now Processing {}".format(file))
        mat_data = scipy.io.loadmat(os.path.join(sourcedir, file))  
        #input = mat_data['lr']#.transpose(2, 0, 1)  # .astype(np.float32)  # .transpose(2, 0, 1)
        label = mat_data['gt'].transpose(2, 0, 1)

        img_HR = torch.tensor(label).float()  # .astype(np.float32)  # .transpose(2, 0, 1)

        C, H, W = img_HR.size()

        prepro = util.SRMDPreprocessing(**degradation_setting)
        LR_img, ker_map, kernels, lr_blured_t, lr_t= prepro(hr_tensor=(img_HR.view(1, C, H, W)), kernel=True, return_blur=True)
        image_LR_blur = LR_img.cpu().numpy().squeeze().astype(np.float32)
        lr_blured_t = lr_blured_t.cpu().numpy().squeeze().astype(np.float32)
        lr_t = lr_t.cpu().numpy().squeeze().astype(np.float32)
        ker_map = ker_map.cpu().numpy().squeeze().astype(np.float32)
        kernels = kernels.cpu().numpy().squeeze().astype(np.float32)
        # Save .mat data
        scipy.io.savemat(os.path.join(savedir, 'blur_{}'.format(file)),
                         {'hr': label, 'lr_blur': image_LR_blur,'lr_blured_t':lr_blured_t,'lr_t':lr_t})


        #data = loadmat(os.path.join(folder, filename))

        # savep = os.path.join(save_path, 'lr_blured_t', file)
        # os.makedirs(savep, exist_ok=True)
        # savep1 = os.path.join(save_path, 'lr_t', file)
        # os.makedirs(savep1, exist_ok=True)
        # for channel in range(C):
        #  
        #     channel_data = lr_blured_t[channel, :, :]
        #     channel_data1 = lr_t[channel, :, :]
        #     channel_data = (
        #             (channel_data - np.min(channel_data)) / (np.max(channel_data) - np.min(channel_data)) * 255).astype(
        #         np.uint8)
        #     channel_data1 = (
        #             (channel_data1 - np.min(channel_data1)) / (
        #                 np.max(channel_data1) - np.min(channel_data1)) * 255).astype(
        #         np.uint8)
        #
        # 
        #     channel_image = Image.fromarray(channel_data)
        #     channel_image1 = Image.fromarray(channel_data1)
        #
        #
        #     channel_image.convert('L').save(os.path.join(savep, f'channel_{channel}.png'), format='png')
        #     channel_image1.convert('L').save(os.path.join(savep1, f'channel_{channel}.png'), format='png')

    print("Image Blurring & Down smaple Done: X" + str(up_scale))


if __name__ == "__main__":
    generate_mod_LR_bic()
