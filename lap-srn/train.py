#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import train_options 
from ops import lrelu
import sys
import os
sys.path.append(os.path.abspath('../'))
import dataset.read_tfrecord
from models import LapSRN
FLAGS = tf.app.flags.FLAGS
def get_train_data(opt):
    filename_queue = tf.train.string_input_producer(
        [opt.tfrecord_dir], num_epochs = opt.num_epochs
        )
    hr_image = dataset.read_tfrecord.read_and_decode(filename_queue, batch_size = opt.batch_size)
    shape = hr_image.get_shape().as_list()
    h = shape[1]
    w = shape[2]
    h = h / pow(2,opt.level)
    w = w / pow(2, opt.level)
    lr_image = tf.image.resize_images(hr_image, [h,w])
    hr_image = tf.cast(hr_image, tf.float32)
    lr_image = tf.cast(lr_image, tf.float32)
    lr_image = lr_image /255.0
    hr_image = hr_image / 255.0
    return hr_image, lr_image

def train(opt):
    with tf.Graph().as_default():

        global_step = tf.get_variable(
            'global_step', [],
            initializer = tf.constant_initializer(0),
            trainable = False
            )
        num_batches_per_epoch = (opt.num_images_epoch /opt. batch_size)
        decay_steps = int(num_batches_per_epoch*1)
        lr = tf.train.exponential_decay(opt.init_lr,
                                       global_step,
                                       10000,
                                       opt.lr_decay_factor,
                                       staircase=True)

        optimizer = tf.train.MomentumOptimizer(lr, momentum = 0.9)
        hr_image, lr_image= get_train_data(opt)
        model_sr = LapSRN()
        loss_op = model_sr.construct_net(lr_image, hr_image)
        train_op = optimizer.minimize(loss_op, global_step = global_step)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver(tf.global_variables()) 
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)
            #tf.train.start_queue_runners(sess = sess)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(opt.log_dir)
            for epoch in range(opt.num_epochs):
                for batch_id in range(num_batches_per_epoch):
                    step = epoch*opt.num_images_epoch + batch_id*opt.batch_size
                    start_time = time.time()
                    _, loss_value = sess.run([train_op, loss_op])
                    if batch_id % 20==0:
                        print(loss_value)
                    if batch_id % 50 == 0:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, batch_id)
                    if step%5000 == 0:
                        checkpoint_path = os.path.join(opt.train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step = step)
            coord.request_stop()
            coord.join(threads)
            
if __name__ == '__main__':
    train(FLAGS)
