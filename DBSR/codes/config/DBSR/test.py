import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils
import scipy.io as scio
from os import listdir
from os.path import join
import numpy as np

import options as option
from models import create_model

try:
    # sys.path.append('..')
    sys.path.append('/home/lwd/code_yp/DBSR-SR')
    import codes.utils as util
    from codes.data import create_dataloader, create_dataset
except ImportError:
    from codes.data import create_dataloader, create_dataset
    pass


from metrics1 import compare_mssim, compare_mpsnr,cal_sam

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
           and "pretrain_model" not in key
           and "resume" not in key
    )
)

# os.system("rm ./result")
# os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["sam"] = []
    test_times = []

    for test_data in test_loader:
        single_img_psnr = []
        single_img_ssim = []
 
        

        #### input dataset_LQ
        model.feed_data(test_data["LQ"], test_data["GT"])
        tic = time.time()
        model.test()
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals()
        sr_img = visuals["SR"].squeeze().numpy()  # uint8

        suffix = opt["suffix"]
        gt_img = visuals["GT"].squeeze().numpy()

        # true_min, true_max = np.min(gt_img), np.max(gt_img)
        # gt_img = (gt_img) / (true_max)
        # pred_min, pred_max = np.min(sr_img), np.max(sr_img)
        # sr_img = (sr_img) / (pred_max)



        sam = cal_sam(sr_img, gt_img)
        psnr = compare_mpsnr(x_true=gt_img, x_pred=sr_img, data_range=1)
        ssim = compare_mssim(x_true=gt_img, x_pred=sr_img, data_range=1, multidimension=False)
        test_results["psnr"].append(psnr)
        test_results["ssim"].append(ssim)
        test_results["sam"].append(sam)
        
        img_name=test_data["name"][0]
        
        logger.info(
            "PSNR: {:.6f} dB; SSIM: {:.6f}; SAM: {:.6f};img: {}".format(
                psnr, ssim,sam,img_name
            )
        )

        SR = sr_img.transpose(1, 2, 0)
        HR = gt_img.transpose(1, 2, 0)
        scio.savemat(dataset_dir +"/"+img_name, {'HR': HR, 'SR': SR})


    ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
    ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
    ave_sam = sum(test_results["sam"]) / len(test_results["sam"])
    logger.info(
        "----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}; SAM: {:.6f}\n".format(
            test_set_name, ave_psnr, ave_ssim,ave_sam
        )
    )
    # if test_results["psnr_y"] and test_results["ssim_y"]:
    #     ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
    #     ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
    #     logger.info(
    #         "----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n".format(
    #             ave_psnr_y, ave_ssim_y
    #         )
    #     )

    print(f"average test time: {np.mean(test_times):.4f}")
