import torch
import numpy as np
import os
import torch.utils.data as data
from PIL import Image
from os import listdir
from os.path import join
import scipy.io as scio


# from scipy.misc import imresize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])


class TrainsetFromFolder(data.Dataset):
    def __init__(self, opt):
        super(TrainsetFromFolder, self).__init__()
        self.opt = opt
        self.image_filenames = [join(opt["dataroot_GT"], x) for x in listdir(opt["dataroot_GT"]) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)

        label = mat['hr'].astype(np.float32)

        # return {"kernels": torch.from_numpy(kernels),"LQ": torch.from_numpy(lr_blur),"lr_blured_t": torch.from_numpy(lr_blured_t),"lr_t": torch.from_numpy(lr_t),"ker_map": torch.from_numpy(ker_map), "lr": torch.from_numpy(lr),"GT": torch.from_numpy(label)}

        # for trainset x8 blur
        return {"GT": torch.from_numpy(label)}

    def __len__(self):
        return len(self.image_filenames)


class ValsetFromFolder(data.Dataset):
    def __init__(self, opt):
        super(ValsetFromFolder, self).__init__()
        self.opt = opt
        self.image_filenames = [join(opt["dataroot_LQGT"], x) for x in listdir(opt["dataroot_LQGT"]) if
                                is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index])
        # input = mat['lr'].astype(np.float32)#.transpose(2, 0, 1)
        input = mat['lr_blur'].astype(np.float32)  # .transpose(2, 0, 1)
        label = mat['hr'].astype(np.float32)  # .transpose(2, 0, 1)
        # print(input.shape)
        # print(label.shape)
        bicu = np.zeros(label.shape, dtype=np.float32)
        imgname = os.path.basename(self.image_filenames[index])

        for i in range(bicu.shape[0]):
            bicu[i, :, :] = scipy_misc_imresize(input[i, :, :], (label.shape[1], label.shape[2]), 'bicubic',
                                                mode=None)  # åŽŸæ¥mode='F',å¯èƒ½è¡¨ç¤ºè¾“å‡ºä¸ºæµ®ç‚¹æ•°

        return {"LQ": torch.from_numpy(input), "GT": torch.from_numpy(label), "Bicu": torch.from_numpy(bicu),
                "name": imgname}

    def __len__(self):
        return len(self.image_filenames)


def scipy_misc_imresize(arr, size, interp='bilinear', mode=None):
    im = Image.fromarray(arr, mode=mode)
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size) * percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size) * size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])  # è°ƒç”¨PILåº“ä¸­çš„resizeå‡½æ•°
    return np.array(imnew)
