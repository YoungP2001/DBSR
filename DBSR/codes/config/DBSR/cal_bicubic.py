import glob
import scipy.io
import numpy as np
import os
import torch
import torch.utils.data as data
from PIL import Image
from os import listdir
from os.path import join
import scipy.io as scio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from eval import PSNR, SSIM, SAM
from metrics1 import compare_mssim, compare_mpsnr,cal_sam

def scipy_misc_imresize(arr, size, interp='bicubic', mode=None):
    im = Image.fromarray(arr, mode=mode)
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size)*percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size)*size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp]) # 调用PIL库中的resize函数
    return np.array(imnew)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])


if __name__ == "__main__":
    #mat_files = glob.glob('D://Hyperspectral//SHSR_code//dataset//Harvard//method2//Test//4_blur_bic_SR//*.mat')
    # 获取文件夹下所有 .mat 文件的路径
    input_path = "/home/lwd/Dataset_yp/HSISR/Chikusei/8/Chikusei_x8_blur2/tests/"
    # out_path = '/home/lwd/code_yp/DBSR-SR/chikusei_bicx8blur/'
    #
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)

    train_psnr1=0
    train_psnr2=0
    ssim1=0
    ssim2=0
    sam=0
    samsum=0

    images_name = [x for x in listdir(input_path) if is_image_file(x)]

    for index in range(len(images_name)):

        mat_data = scio.loadmat(input_path + images_name[index])

        input = mat_data['lr_blur'].astype(np.float32)#.transpose(2, 0, 1)
        label = mat_data['hr'].astype(np.float32)#.transpose(2, 0, 1)
        bicu = np.zeros(label.shape, dtype=np.float32)

        for i in range(bicu.shape[0]):
            bicu[i, :, :] = scipy_misc_imresize(input[i, :, :], (label.shape[1], label.shape[2]), 'bicubic', mode=None) # 原来mode='F',可能表示输出为浮点数
        # label = mat_data['HR'].astype(np.float32).transpose(2, 0, 1)
        # bicu = mat_data['SR'].astype(np.float32).transpose(2, 0, 1)


        # true_min, true_max = np.min(label), np.max(label)
        # label = (label - true_min) / (true_max - true_min)
        # pred_min, pred_max = np.min(bicu), np.max(bicu)
        # bicu = (bicu - pred_min) / (pred_max - pred_min)

        p1= compare_mpsnr(label, bicu, data_range=1)
        train_psnr1 += p1
        # p2= PSNR(bicu, label)
        # train_psnr2 +=p2
        print("p1:", p1)
        # print("p2:",p2)
        s1=compare_mssim(label, bicu, data_range=1, multidimension=False)
        ssim1 += s1
        # s2=SSIM(bicu, label)
        # ssim2 += s2
        print("s1:", s1)
        # print( "s2:", s2)

        sam=cal_sam(bicu, label)
        samsum += sam
        print("sam:", sam)

        SR = bicu.transpose(1, 2, 0)
        HR = label.transpose(1, 2, 0)
        # scio.savemat(out_path +"/"+ images_name[index], {'HR': HR, 'SR': SR})
    # print("avg_train_psnr:", "   ",train_psnr2/len(images_name))
    # print("avg_train_ssim:","   " ,ssim2/len(images_name))
    print("avg_train_psnr1:", "   ",train_psnr1/len(images_name))
    print("avg_train_ssim1:","   " ,ssim1/len(images_name))
    print("avg_train_sam:", samsum/len(images_name))
    print("len:", len(images_name))
# compare_mpsnr compare_mssim
# -------------
# no blur
# avg_train_psnr: 36.56986531416376
# avg_train_ssim: 0.9439718156203553

# blur
# avg_train_psnr:33.323
# avg_train_ssim:0.911

# PSNR SSIM
# -------------
# no blur
# avg_train_psnr:
# avg_train_ssim:

# blur
# avg_train_psnr:
# avg_train_ssim:
# avg_train_psnr: 33.537318590366226     32.9484096462325
# avg_train_ssim: 0.912724129387423     0.913096986771111

# blur2 4x
# avg_train_psnr: 31.727766420030473     31.36718725734345
# avg_train_ssim: 0.9042650901692717     0.9074841101002948
# avg_train_sam: 4.131867781064274
