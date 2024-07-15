import argparse
import logging
import math
import os
import random
import sys
import copy
sys.path.append("/home/lwd/code_yp/DBSR")  
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# from IPython import embed

import options as option
from models import create_model



import codes.utils as util
from codes.data import create_dataloader, create_dataset
from codes.data.data_sampler import DistIterSampler

from codes.data.data_util import bgr2ycbcr
from eval import PSNR, SSIM, SAM
from metrics1 import compare_mssim, compare_mpsnr,cal_sam


def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
            mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)


    print("load PCA matrix")
    pca_matrix = torch.load(
        opt["pca_matrix_path"], map_location=lambda storage, loc: storage
    )
    print("PCA matrix shape: {}".format(pca_matrix.shape))

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        util.set_random_seed(opt['train']['manual_seed'])

    torch.backends.cudnn.benchmark = True  
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),  # ???cpu???,???GPU
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                       and "pretrain_model" not in key
                       and "resume" not in key
                )
            )
            os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            # my trainsize 800, batch size =64, iter=13,???train_size?????epoch??iter,??13
            total_iters = int(opt["train"]["niter"])  # niter???? 500000
            total_epochs = int(math.ceil(total_iters / train_size))  # ?????????iter,???13??epoch
            # total_epochs =10
            print("train_size:", math.ceil(len(train_set)))
            print("total_iters:", total_iters)

            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None, "train_loader is None"
    assert val_loader is not None, "val_loader is None"

    #### create model
    model = create_model(opt)  # load pretrained model of SFTMD

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0


    prepro = util.SRMDPreprocessing(
        scale=opt["scale"], pca_matrix=pca_matrix, cuda=True, **opt["degradation"]
    )
    kernel_size = opt["degradation"]["ksize"]
    padding = kernel_size // 2


    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_psnr = 0.0
    best_psnr1 = 0.0
    best_iter = 0
    best_iter1 = 0
    i = 0
    # if rank <= 0:
    prev_state_dict = copy.deepcopy(model.netG.module.state_dict())
    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        # ??iteration
        for iteration, train_data in enumerate(train_loader):
            num_images = 0
            total_psnr = 0
            current_step += 1
            
            print("current_step:", current_step)
            if current_step > total_iters:
                break

            # train_data?tensor??
            GT_img=train_data["GT"]
            LR_img=train_data["LQ"]
            #ker_map=train_data["ker_map"]
            #kernels=train_data["kernels"]
            lr_blured_t=train_data["lr_blured_t"]
            lr_t=train_data["lr_t"]
            #LR_img, ker_map, kernels, lr_blured_t, lr_t = prepro(GT_img, True, return_blur=True)


            model.feed_data(
                LR_img, GT_img, lr_blured=lr_blured_t, lr=lr_t)
            model.optimize_parameters(current_step)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

            visuals = model.get_current_visuals()
            Train_SR = visuals["Batch_SR"].numpy()
            Train_HR = visuals["Batch_GT"].numpy()
            #print("Train_SR:", Train_SR.shape)
            #print("Train_HR:", Train_HR.shape)
            #print("len(Train_HR):", len(Train_HR))
            for n in range(len(Train_HR)):
                train_psnr = compare_mpsnr(Train_HR[n], Train_SR[n], data_range=1)
                total_psnr += train_psnr
                num_images += 1
            avg_train_psnr = total_psnr / num_images
            print("avg_train_psnr:", avg_train_psnr)
            logger.info(
                "# Train # AVG_PSNR1: {:.6f}, Iter: {}".format(avg_train_psnr, current_step))


            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                # k:key,v:value,??? item
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank == 0:
                    logger.info(message)

            # validation, to produce ker_map_list(fake)
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                val_psnr = 0
                val_ssim = 0
                val_psnr1 = 0
                val_ssim1 = 0
                val_sam = 0
                idx = 0
                print(val_loader)
                for _, val_data in enumerate(val_loader):
                    # LR_img, ker_map = prepro(val_data['GT'])
                    LR_img = val_data["LQ"]
                    bic_img=val_data["Bicu"].float().cpu().numpy()

                    # valid Predictor
                    model.feed_data(LR_img, val_data["GT"])
                    model.test()
                    visuals = model.get_current_visuals()
                    SR = visuals["SR"].numpy()
                    HR = visuals["GT"].numpy()

                    p2=compare_mpsnr(x_true=HR, x_pred=SR, data_range=1)
                    #val_psnr += p1
                    val_psnr1 += p2
                    #val_ssim += SSIM(SR, HR)
                    val_ssim1 += compare_mssim(x_true=HR, x_pred=SR, data_range=1, multidimension=False)
                    val_sam += cal_sam(SR, HR)
                    idx += 1
                    print("idx:",idx,"PSNR1",p2)
                print("idx:",idx,"val_psnr1",val_psnr1)

                #avg_psnr = val_psnr / idx
                #avg_ssim = val_ssim / idx
                avg_psnr1 = val_psnr1 / idx
                avg_ssim1 = val_ssim1 / idx
                avg_sam = val_sam / idx
                # if avg_psnr > best_psnr:
                    #best_psnr = avg_psnr
                    #best_iter = current_step
                if avg_psnr1 > best_psnr1:
                    best_psnr1 = avg_psnr1
                    best_iter1 = current_step
                #print("avg_psnr:",avg_psnr,"avg_ssim:",avg_ssim)
                print("avg_psnr:",avg_psnr1,"avg_ssim:",avg_ssim1,"avg_sam:",avg_sam)
                # log
                #logger.info(
                #   "# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}, SSIM: {:.6f}".format(avg_psnr, best_psnr, best_iter, avg_ssim))
                logger.info(
                    "# Validation # PSNR1: {:.6f}, Best PSNR1: {:.6f}| Iter1: {}, SSIM: {:.6f},SAM: {:.6f}".format(avg_psnr1,
                                                                                                       best_psnr1,
                                                                                                       best_iter1,
                                                                                                       avg_ssim1,
                                                                                                       avg_sam))
                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}, psnr1: {:.6f}, ssim1: {:.6f},SAM: {:.6f}".format(
                        epoch, current_step, avg_psnr1, avg_ssim1,avg_sam
                    )
                )
                # tensorboard logger
                if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    tb_logger.add_scalar("psnr", avg_psnr1, current_step)

                if avg_psnr1 > 20:
                    # if rank <= 0:
                    prev_state_dict = copy.deepcopy(model.netG.module.state_dict())
                    # torch.save(prev_state_dict, opt["name"]+".pth")
                else:
                    logger.info("# Validation crashed, use previous state_dict...\n")
                    model.netG.module.load_state_dict(copy.deepcopy(prev_state_dict), strict=True)
                    # model.netG.module.load_state_dict(torch.load(opt["name"]+".pth"), strict=True)
                    # model.load_network(opt["name"]+".pth", model.netG)
                    # break

            #### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0 and avg_psnr1 > 30:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)
            i += 1
    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()


if __name__ == "__main__":
    main()
