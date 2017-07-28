#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
from models import LapSRN
from utils import imgread, imshow
from metric import psnr
def demo(lr_image, hr_image):
    model_sr = LapSRN(mode = 'demo')
    hr_images_fake, residuals = model_sr.construct_net(lr_image, hr_image)
    ckpt_path = tf.train.latest_checkpoint('checkpoint')
    print(ckpt_path)
    restorer = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        restorer.restore(sess, ckpt_path)
        hr_image_fake_level_2 = hr_images_fake['hr_image_fake_level_1']+residuals['residual_level_1']
        hr_image_fake_level_2 = tf.clip_by_value(hr_image_fake_level_2, 0, 1)
        hr_image_fake_level_2 = sess.run(hr_image_fake_level_2)
        hr_image_fake_level_2 = hr_image_fake_level_2.squeeze()
        lr_image = sess.run(lr_image)
        lr_image = lr_image.squeeze()
        hr_image = sess.run(hr_image)
    psnr_value = psnr(hr_image.squeeze(), hr_image_fake_level_2.squeeze())
    print(psnr_value)
    imshow(hr_image.squeeze())
    imshow(hr_image_fake_level_2)
    
if __name__ == '__main__':
    img_lr, img = imgread('data/8049.png')
    demo(img_lr, img)
