from math import exp
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision import models

class build_metric(nn.Module):
    def __init__(self):
        super().__init__()

        self.L1Loss = nn.L1Loss()
        # LPIPS
        self.lpips_model = LPIPS(net="alex", verbose=False)

        # freeze
        self.eval()
        self.requires_grad_(False)

    def forward(self, gt, pred=None):
        # inputs should be [0,1] here
        assert gt.shape[0] == pred.shape[0]
        bsz = gt.shape[0]

        # LPIPS
        lpips = self.lpips_model(pred, gt, normalize=True).reshape(bsz, -1)
        l1 = self.L1Loss(pred, gt)
        # PSNR & SSIM
        img_gts = gt.cpu().numpy()
        img_preds = pred.cpu().numpy()
        psnr = []
        ssim = []
        ssim_256 = []

        for i in range(bsz):
            img_gt = img_gts[i]
            img_pred = img_preds[i]

            psnr.append(compare_psnr(img_gt, img_pred, data_range=1))
            ssim.append(compare_ssim(img_gt, img_pred, data_range=1, win_size=51, channel_axis=0))

            img_gt_256 = img_gt * 255.0
            img_pred_256 = img_pred * 255.0
            ssim_256.append(compare_ssim(img_gt_256, img_pred_256, gaussian_weights=True, sigma=1.5,
                                         use_sample_covariance=False, channel_axis=0,
                                         data_range=img_pred_256.max() - img_pred_256.min()))

        psnr = torch.tensor(psnr).to(gt.device).reshape(bsz, -1)
        ssim = torch.tensor(ssim).to(gt.device).reshape(bsz, -1)
        ssim_256 = torch.tensor(ssim_256).to(gt.device).reshape(bsz, -1)
        return lpips, psnr, ssim, ssim_256,l1


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)