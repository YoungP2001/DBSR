import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as func
from codes.utils import BatchBlur, b_Bicubic, normkernel_to_downkernel, zeroize_negligible_val


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = torch.add(x, -y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == "gan" or self.gan_type == "ragan":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_type == "wgan-gp":

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError(
                "GAN type [{:s}] is not found".format(self.gan_type)
            )

    def get_target_label(self, input, target_is_real):
        if self.gan_type == "wgan-gp":
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer("grad_outputs", torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(
            outputs=interp_crit,
            inputs=interp,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss


# k_pred??????????;lr_blured?lr???????????????????????????????????
class CorrectionLoss(nn.Module):
    def __init__(self, scale=4.0, eps=1e-6):
        super(CorrectionLoss, self).__init__()
        self.scale = scale
        self.eps = eps
        self.cri_pix = nn.L1Loss()

    def forward(self, k_pred, lr_blured, lr):

        ks = []
        mask = torch.ones_like(k_pred).cuda()
        for c in range(lr_blured.shape[1]):
            k_correct = normkernel_to_downkernel(lr_blured[:, c:c+1, ...], lr[:, c:c+1, ...], k_pred.size(), self.eps)
            ks.append(k_correct.clone())
            mask *= k_correct
        ks = torch.cat(ks, dim=1)
        k_correct = torch.mean(ks, dim=1, keepdim=True) * (mask>0)
        k_correct = zeroize_negligible_val(k_correct, n=40) #k_correct = zeroize_negligible_val(k_correct, n=40)

        return self.cri_pix(k_pred, k_correct), k_correct



class myLoss(nn.Module):
    def __init__(self, sam_lamd=1e-1, spatial_tv=False, spectral_tv=False,epoch=None):
        super(myLoss, self).__init__()
        self.sam_lamd = sam_lamd
        self.use_spatial_TV = spatial_tv
        self.use_spectral_TV = spectral_tv
        self.epoch=epoch
        #self.fidelity = torch.nn.L1Loss()
        self.fidelity = CharbonnierLoss()
        # self.spatial = TVLoss(weight=1e-3)
        # self.spectral = TVLossSpectral(weight=1e-3)
        self.loss_sam=Loss_Sam(self.sam_lamd, epoch=self.epoch)
        self.loss_sad=reconstruction_SADloss()

    def forward(self, y, gt):
        loss = self.fidelity(y, gt)
        # spatial_TV = 0.0
        # spectral_TV = 0.0
        # loss_sam=0.0
        if self.use_spatial_TV:
            spatial_TV = self.spatial(y)
        if self.use_spectral_TV:
            spectral_TV = self.spectral(y)

        loss_sam = self.loss_sam(y, gt)
        loss_sad=self.loss_sad(y, gt)

        total_loss = loss + 0.1*loss_sad
        return total_loss


# from https://github.com/jxgu1016/Total_Variation_Loss.pytorch with slight modifications
class TVLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        # h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]).sum()
        # w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]).sum()
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TVLossSpectral(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLossSpectral, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        c_x = x.size()[1]
        count_c = self._tensor_size(x[:, 1:, :, :])
        # c_tv = torch.abs((x[:, 1:, :, :] - x[:, :c_x - 1, :, :])).sum()
        c_tv = torch.pow((x[:, 1:, :, :] - x[:, :c_x - 1, :, :]), 2).sum()
        return self.TVLoss_weight * 2 * (c_tv / count_c) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class reconstruction_SADloss(torch.nn.Module):
    def __init__(self):
        super(reconstruction_SADloss, self).__init__()

    def forward(self, x, y):
        abundance_loss = torch.acos(torch.cosine_similarity(x, y, dim=1))
        abundance_loss = torch.mean(abundance_loss)
        return abundance_loss


class Loss_Sam(nn.Module):
    def __init__(self, lamd = 1e-1, epoch=None):
        super(Loss_Sam, self).__init__()
        #self.N = N
        self.lamd = lamd
        # self.mse_lamd = mse_lamd
        self.epoch = epoch
        return

    def forward(self, res, label):
        #mse = func.mse_loss(res, label, size_average=False)
        # mse = func.l1_loss(res, label, size_average=False)
        #loss = mse / (self.N * 2)
        esp = 1e-12
        N = label.size()[0]
        H = label.size()[2]
        W = label.size()[3]
        Itrue = label.clone()
        Ifake = res.clone()
        nom = torch.mul(Itrue, Ifake).sum(dim=1)
        denominator = Itrue.norm(p=2, dim=1, keepdim=True).clamp(min=esp) * \
                      Ifake.norm(p=2, dim=1, keepdim=True).clamp(min=esp)
        denominator = denominator.squeeze()
        # sam = -np.pi/2*torch.div(nom, denominator) + np.pi/2
        sam = torch.div(nom, denominator).acos()
        sam[sam!=sam] = 0
        sam_sum = torch.sum(sam) / (N * H * W)
        if self.epoch is None:
            total_loss =  self.lamd * sam_sum  # + self.mse_lamd * loss
        else:
            # norm = self.mse_lamd + self.lamd * 0.1 **(self.epoch//10)
            lamd_sam = self.lamd * 0.1 ** (self.epoch // 10)
            total_loss = lamd_sam * sam_sum # + self.mse_lamd/norm * loss
        return total_loss