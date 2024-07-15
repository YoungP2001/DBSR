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
#import utils.util.SRMDPreprocessing as SRMD

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

    # savedir = "D://Hyperspectral//SHSR_code//dataset//CAVE//method2//Val//32x32//4_blur2//"
    # sourcedir = "D://Hyperspectral//SHSR_code//dataset//CAVE//method2//Val//32x32//4//"
    # savedir = "/data/yp/HSI/Data/Cave/Test/8_blur/"
    # sourcedir = "/data/yp/HSI/Data/Cave/Test/8/"
    savedir = "/home/lwd/Dataset_yp/HSISR/Chikusei/8/Chikusei_x8_blur/tests"
    sourcedir = "/home/lwd/Dataset_yp/HSISR/Chikusei/4/Chikusei_x4/tests/"
    # mat_files = glob.glob(sourcedir)  # 获取文件夹下所有 .mat 文件的路径

    if not os.path.exists(savedir):
        os.makedirs(savedir)
        print("Folder created")
    else:
        print("Folder already exists")

    # load PCA matrix of enough kernel

    print("load PCA matrix")
    pca_matrix = torch.load(
        "../../pca_matrix/DBSR/pca_matrix.pth", map_location=lambda storage, loc: storage
    )
    print("PCA matrix shape: {}".format(pca_matrix.shape))

    degradation_setting = {
        "random_kernel": False,
        "code_length": 10,
        "ksize": 21,
        "pca_matrix": pca_matrix,
        "scale": up_scale,
        "cuda": True,
        "rate_iso": 1.0
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
    #     "sig_min": 2.0,
    #     "sig_max": 5
    # }


    files = os.listdir(sourcedir)
    # num_files_to_pick = 100
    # selected_files = random.sample(files, num_files_to_pick)
    for file in files:
        print("Now Processing {}".format(file))
        mat_data = scipy.io.loadmat(os.path.join(sourcedir, file))  # 读取 .mat 文件的数据

        #input = mat_data['ms'].astype(np.float32)  # .astype(np.float32)  # .transpose(2, 0, 1)
        label = mat_data['gt'].astype(np.float32).transpose(2, 0, 1)
        img_HR = torch.tensor(label).float()  # .astype(np.float32)  # .transpose(2, 0, 1)
        C, H, W = img_HR.size()

        for sig in np.linspace(1.8, 3.2, 8):
            prepro = util.SRMDPreprocessing(sig=sig, **degradation_setting)
            LR_img, ker_map, kernels, lr_blured_t, lr_t= prepro(hr_tensor=(img_HR.view(1, C, H, W)), kernel=True, return_blur=True)
            image_LR_blur = LR_img.cpu().numpy().squeeze().astype(np.float32)
            lr_blured_t = lr_blured_t.cpu().numpy().squeeze().astype(np.float32)
            lr_t = lr_t.cpu().numpy().squeeze().astype(np.float32)
            #ker_map = ker_map.cpu().numpy().squeeze().astype(np.float32)
            kernels = kernels.cpu().numpy().squeeze().astype(np.float32)
            # Save .mat data
            scipy.io.savemat(os.path.join(savedir, 'blur_{}_{}'.format(sig,file)),
                             {'hr': label, 'lr_blur': image_LR_blur,'kernels':kernels})

            # # 保存为图像
            # savep = os.path.join(img_save_path, 'lr_blured_t', '{}_{}'.format(file,i))
            # os.makedirs(savep, exist_ok=True)
            # savep1 = os.path.join(img_save_path, 'lr_t', '{}_{}'.format(file,i))
            # os.makedirs(savep1, exist_ok=True)
            # for channel in range(C):
            #     # 取出对应通道的数据
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
            #     # 将数组转换为图像
            #     channel_image = Image.fromarray(channel_data)
            #     channel_image1 = Image.fromarray(channel_data1)
            #
            #     # 保存图像文件
            #
            #     channel_image.convert('L').save(os.path.join(savep, f'channel_{channel}.png'), format='png')
            #     channel_image1.convert('L').save(os.path.join(savep1, f'channel_{channel}.png'), format='png')

    print("Image Blurring & Down smaple Done: X" + str(up_scale))


if __name__ == "__main__":
    generate_mod_LR_bic()
