import tensorflow as tf
import tensorflow.contrib.slim as slim
from ops import lrelu, upsample_layer, sparsity_summary
import sys
import os
sys.path.append(os.path.abspath('../'))
import dataset
class LapSRN():
    def __init__(self, level = 2, nfc = 64, depth = 5, mode = 'train',  scope = 'LapSRN'):
        self.scope = scope
        self.level = level
        self.depth = depth
        self.nfc = nfc
        self.mode = mode
        with tf.variable_scope(scope) as scope:
            self.train = tf.placeholder(tf.bool)
            #self.construct_net()

    def feature_extract_net(self, lr_image):
        end_points = {}
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], 
                           activation_fn = lrelu,
                           ):
            conv = slim.conv2d(lr_image, self.nfc, [3,3], scope = 'conv1') 
            for l in range(self.level):
                for d in range(self.depth):
                    conv = slim.conv2d(conv, self.nfc, [3,3], scope = 'conv_%d_level_%d'%(l,d))
                conv = slim.conv2d_transpose(conv, self.nfc, [4,4], stride = 2, scope = 'residual_level_%d'%(l))
                conv = slim.conv2d(conv, 3, [3,3], activation_fn = None, scope = 'conv_level_%d'%(l))
                end_points['residual_level_%d'%(l)] = conv
        return end_points 
    def bilinear_net(self, lr_image):
        hr_images = {}
        shape = lr_image.get_shape().as_list()
        h = shape[1]
        w = shape[2]
        #shape = [shape[1], shape[2]]
        hr_image = lr_image
        for l in range(self.level):
            hr_image = upsample_layer(hr_image, mode = 'bilinear')
            hr_images['hr_image_fake_level_%d'%(l)] = hr_image
        return hr_images
    def get_hr_images(self, hr_image):
        hr_images = {}
        shape = hr_image.get_shape().as_list()
        w = shape[1]
        h = shape[2]
        for l in range(self.level-1, -1, -1):
            if l != self.level-1:
                w = w/2
                h = h/2
            hr_image = tf.image.resize_images(hr_image, [w,h])
            hr_images['hr_image_gt_level_%d'%(l)] = hr_image
        return hr_images
    def construct_net(self, lr_image, hr_image):
        residuals = self.feature_extract_net(lr_image)
        hr_images_fake = self.bilinear_net(lr_image)
        hr_images_gt = self.get_hr_images(hr_image)
        if self.mode == 'demo':
            return hr_images_fake, residuals
        loss = 0
        for l in range(self.level):
            hr_image_fake = hr_images_fake['hr_image_fake_level_%d'%(l)]
            hr_image_gt = hr_images_gt['hr_image_gt_level_%d'%(l)]
            print(hr_image_gt.dtype)
            hr_image_gt = tf.cast(hr_image_gt, tf.float32)
            tf.summary.image('hr_gt_level_%d'%(l), hr_image_gt)
            residual = residuals['residual_level_%d'%(l)]
            sparsity_summary(residual, 'residual_level_%d'%(l))
            tf.summary.image('residual_level_%d'%(l), residual)
            hr_image_fake = hr_image_fake + residual
            tf.summary.image('hr_fake_level_%d'%(l), hr_image_fake)
            loss  = loss + self.reconstruction_loss(hr_image_fake, hr_image_gt)
        if self.mode == 'train':
            return loss
    def reconstruction_loss(self, gt, fake):
        eps = tf.ones(gt.get_shape())
        eps = 1e-6*eps
        loss = tf.reduce_mean(tf.sqrt(tf.square(fake-gt)+eps))
        return loss
def test_LapSrn():
    model_sr = LapSRN()
    hr_image = tf.ones((20, 128, 128, 3))
    lr_image = tf.ones((20, 32, 32, 3))
    loss = model_sr.construct_net(lr_image, hr_image)
    init_op = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        loss_value = sess.run(loss)
        print(loss_value)

if __name__=='__main__':
    test_LapSrn()

