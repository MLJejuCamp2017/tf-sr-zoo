import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import tensorflow as tf
import skimage
import skimage.color
import math
def imshow(img):
    plt.imshow(img)
    plt.show()

def imgread(img_path, scale = 4):
    img = scipy.misc.imread(img_path)
    #img = scipy.misc.imresize(img, (128, 128))
    img = img /256.0
    h,w,c = img.shape
    new_h = pow(2, int(math.log(h, 2))+1)
    tmp1 = new_h - h 
    new_w = pow(2, int(math.log(w, 2))+1)
    tmp2 = new_w - w
    img = np.pad(img, ((0,tmp1), (0, tmp2),(0,0)), mode = 'constant')
    if scale != None:
        img = np.expand_dims(img,0)
        img = tf.convert_to_tensor(img)
        lr_w = new_w / scale
        lr_h = new_h /scale
        img = tf.cast(img, tf.float32)
        img_lr = tf.image.resize_images(img, [lr_h, lr_w])
        img_lr = tf.cast(img_lr,tf.float32)
        return img_lr, img
    return img
def rgb2gray(img):
    print img.shape
    img = skimage.color.rgb2gray(img)
    return img
if __name__ == '__main__':
    img = imgread('data/emma.jpg')
    img = img.squeeze()
    imshow(img)

