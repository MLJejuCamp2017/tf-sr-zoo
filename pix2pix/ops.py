import tensorflow as tf
slim = tf.contrib.slim

def conv2d(input, out_channels, scope = None):
	padded_input = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
	conv = slim.conv2d(padded_input, out_channels, [3,3], scope = scope)
	return conv
def lrelu(x, alpha = 0.2, name = None):
    return tf.maximum(x, alpha*x)

def upsample_layer(x, out_channels, scope = None, scale = 2,  mode = 'bilinear'):
    if mode == 'deconv':
        conv = slim.conv2d_transpose(x, out_channels, [4,4], stride = scale, activation_fn = lrelu, scope = scope)
        #conv = slim.conv2d(conv, 3,[3,3], activation_fn = None)
        return conv
    if mode == 'bilinear':
        shape = x.get_shape().as_list()
        h = shape[1]
        w = shape[2]
        conv = tf.image.resize_images(x, (scale*h, scale*w))
        return conv