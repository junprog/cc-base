# TODO: mae, mse, nae, game
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def game(output, target, L=0):
    # eg: L=3, p=8 p^2=64
    p = pow(2, L)
    abs_error = 0
    square_error = 0
    _, _, H, W = target.shape
    for i in range(p):
        for j in range(p):
            # print i, j, (i*H/p,(i+1)*H/p), (j*W/p,(j+1)*W/p)
            output_block = output[:, :, i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]
            target_block = target[:, :, i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]
            abs_error += abs(output_block.cpu().data.sum()-target_block.cpu().data.sum().float())
            square_error += (output_block.cpu().data.sum()-target_block.cpu().data.sum().float()).pow(2)

    return abs_error, square_error

def psnr(output, target):
    mse = torch.mean((output - target) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))

def ssim(output, target):
    pass