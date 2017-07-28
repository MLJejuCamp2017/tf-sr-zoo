#!/usr/bin/env python
# coding=utf-8
import options
import tensorflow as tf
import os
FLAGS = tf.app.flags.FLAGS

class imdb():
    def __init__(self, opt):
        self.opt = opt
        self.get_image_files()
    def get_image_files(self):
        opt = self.opt
        datasets = opt.datasets.split(',')
        img_files = []
        for dataset in datasets:
            dataset_path = os.path.join(opt.datasets_dir, dataset)
            img_list = os.path.join('lists', dataset+'.txt')
            img_list = open(img_list, 'r').readlines()
            for img_name in img_list:
                img_name = img_name.strip()
                img_path = os.path.join(dataset_path, img_name+'.png')
                img_path = os.path.abspath(img_path)
                img_files.append(img_path)
                print img_path
        self.img_files = img_files
if __name__ == '__main__':
    img_files = imdb(FLAGS)
    img_files.get_image_files()

