#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
import scipy
import scipy.misc
from utils import rgb2gray 
def mse(img_fake, img_gt):
    h, w = img_fake.shape
    resiudal = img_fake - img_gt
    loss = np.mean(np.square(img_fake-img_gt))
    return loss
def psnr(img_fake, img_gt, is_color = True):
    img_fake_gray = rgb2gray(img_fake)
    img_gt_gray = rgb2gray(img_gt)
    mse_loss = mse(img_fake_gray, img_gt_gray)
    psnr_value = 10*np.log10(1.0/mse_loss)
    return psnr_value
