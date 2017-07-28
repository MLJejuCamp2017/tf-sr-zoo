#!/usr/bin/env python
# coding=utf-8
import options
import random
import tensorflow as tf
from imdb import imdb
from PIL import Image
import numpy as np
FLAGS = tf.app.flags.FLAGS



def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
def gen_tfrecord(imdb):
    num_patchs = 0
    patch_size = FLAGS.patch_size
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    for num in range(FLAGS.epoch):
        for img_file in imdb.img_files:
            #img = np.array(Image.open(img_file))
            img = Image.open(img_file)
            height = img.size[0]
            width = img.size[1]
            if height > FLAGS.patch_size and width > FLAGS.patch_size:
                end_h = height - FLAGS.patch_size
                end_w = width - FLAGS.patch_size
                start_h = random.randint(0, end_h)
                start_w = random.randint(0, end_w)
                img = img.crop((start_h, start_w, start_h+patch_size, start_w+patch_size))
                img_raw = img.tobytes()
                example = tf.train.Example(features = tf.train.Features(feature={
                    'height': _int64_feature(patch_size),
                    'width': _int64_feature(patch_size),
                    'hr_image': _bytes_feature(img_raw),
                }))
                writer.write(example.SerializeToString())
            else:
                continue
    writer.close()

if __name__ == '__main__':
    imdb = imdb(FLAGS)
    gen_tfrecord(imdb)
