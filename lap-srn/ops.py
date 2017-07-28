#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
def lrelu(x, alpha = 0.2, name = None):
    return tf.maximum(x, alpha*x)

def upsample_layer(x, scope = None, scale = 2,  mode = 'bilinear'):
    if mode == 'deconv':
        conv = slim.conv2d_transpose(x, 64, [4,4], stride = scale, activation_fn = lrelu, scope = scope)
        conv = slim.conv2d(conv, 3,[3,3], activation_fn = None)
        return conv
    if mode == 'bilinear':
        shape = x.get_shape().as_list()
        h = shape[1]
        w = shape[2]
        conv = tf.image.resize_images(x, (scale*h, scale*w))
        conv = slim.conv2d(conv, 3, [1,1], activation_fn = None)
        return conv

def sparsity_summary(x,name = None):
    zero_fraction = tf.nn.zero_fraction(x)
    tf.summary.scalar(name, zero_fraction)

