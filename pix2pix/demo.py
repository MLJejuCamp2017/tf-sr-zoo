from models import pix2pix_model
import tensorflow as tf
from cfg import cfg
from utils import imgread, imshow
from metric import psnr
def demo(img_path):
	lr_img, hr_img = imgread(img_path)
	model = pix2pix_model(cfg)
	model.test_model(lr_img, hr_img)
	ckpt_path = tf.train.latest_checkpoint('checkpoint')
	restorer = tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		restorer.restore(sess, ckpt_path)
		hr_image_fake = model.fake_hr_image
		hr_image_fake = tf.clip_by_value(hr_image_fake, 0, 1)
		hr_image_fake = sess.run(hr_image_fake)
		hr_image_fake = hr_image_fake.squeeze()
		hr_image = sess.run(hr_img)
	psnr_value = psnr(hr_image.squeeze(), hr_image_fake.squeeze())
	print(psnr_value)
	imshow(hr_image_fake)
	imshow(hr_image.squeeze())

if __name__ == '__main__':
	demo('data/emma.jpg')
