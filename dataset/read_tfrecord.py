#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

def features():
    feature = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'hr_image': tf.FixedLenFeature([], tf.string)
    }
    return feature

def read_and_decode(filename_queue, batch_size):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    feature = features()
    feature = tf.parse_single_example(
        serialized_example,
        features = feature,
        )
    hr_image = tf.decode_raw(feature['hr_image'], tf.uint8)
    height = tf.cast(feature['height'], tf.int32)
    width = tf.cast(feature['width'], tf.int32)
    print(height)
    image_shape = tf.stack([128, 128,3 ])
    hr_image = tf.reshape(hr_image, image_shape)
    hr_image = tf.image.random_flip_left_right(hr_image)
    hr_image = tf.image.random_contrast(hr_image, 0.5, 1.3)
    hr_images = tf.train.shuffle_batch([hr_image], batch_size = batch_size, capacity = 30,
                                      num_threads = 2,
                                        min_after_dequeue = 10)
    return hr_images
    

if __name__ == '__main__':
    filename_queue = tf.train.string_input_producer(
        ['tfrecord/train.record'], num_epochs=1
    )
    hr_images = read_and_decode(filename_queue, batch_size = 20)
    init_op = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer())
    tf.summary.image('hr_images', hr_images)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/image')
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        for i in xrange(10):
            summary = sess.run(merged)
            writer.add_summary(summary)
        coord.request_stop()
        coord.join(threads)
