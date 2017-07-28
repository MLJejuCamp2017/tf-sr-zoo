import tensorflow as tf
import sys
import os
from models import pix2pix_model
from cfg import cfg
sys.path.append(os.path.abspath('../'))
import dataset.read_tfrecord
def get_train_data(opt):
    filename_queue = tf.train.string_input_producer(
        [opt.tfrecord_dir], num_epochs = opt.num_epochs
        )
    hr_image = dataset.read_tfrecord.read_and_decode(filename_queue, batch_size = opt.batch_size)
    shape = hr_image.get_shape().as_list()
    h = shape[1]
    w = shape[2]
    h = h / 4
    w = w / 4
    lr_image = tf.image.resize_images(hr_image, [h,w])
    hr_image = tf.cast(hr_image, tf.float32)
    lr_image = tf.cast(lr_image, tf.float32)
    lr_image = lr_image /255.0
    hr_image = hr_image / 255.0
    return hr_image, lr_image

def train(cfg):
	with tf.Graph().as_default():
		hr_image, lr_image = get_train_data(cfg)
		model = pix2pix_model(cfg)
		model.construct_model(lr_image, hr_image)
		global_step = tf.get_variable(
            'global_step', [],
            initializer = tf.constant_initializer(0),
            trainable = False
            )
		incr_global_step = tf.assign(global_step, global_step+1)
		train_op = [model.g_train, model.d_train, model.g_loss, model.d_loss, incr_global_step]
		### add summary
		tf.summary.image("lr_image", lr_image)
		tf.summary.image("bilinear", model.bilinear_hr_image)
		tf.summary.image("fake", model.fake_hr_image)
		tf.summary.image("hr_image", hr_image)

		tf.summary.scalar("d_loss", model.d_loss)
		tf.summary.scalar("g_loss", model.g_loss)
		tf.summary.scalar("g_l1_loss", model.g_l1_loss)


		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		saver = tf.train.Saver(tf.global_variables(), max_to_keep = 1)

		with tf.Session() as sess:
			sess.run(init_op)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord = coord)
			summary_op = tf.summary.merge_all()
			summary_writer = tf.summary.FileWriter(cfg.log_dir)
			for epoch in range(cfg.num_epochs):
				for batch_id in range(cfg.num_batches_per_epoch):
					#step = epoch*cfg.num_images_epoch + batch_id*cfg.batch_size
					loss = sess.run(train_op)
					d_loss = loss[3]
					g_loss = loss[2]
					if batch_id % 20==0:
						print(d_loss, g_loss)
					if batch_id % 50 == 0:
						summary_str = sess.run(summary_op)
						summary_writer.add_summary(summary_str, batch_id)
					if batch_id%5000 == 0:
						checkpoint_path = os.path.join(cfg.train_dir, 'model.ckpt')
						saver.save(sess, checkpoint_path, global_step = global_step)
			coord.request_stop()
			coord.join(threads)
if __name__ == '__main__':
	train(cfg)

