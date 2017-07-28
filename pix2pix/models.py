import tensorflow as tf
slim = tf.contrib.slim
from ops import lrelu, upsample_layer, conv2d
eps = 1e-12


def create_generator(hr_image_bilinear, num_channels, cfg):
	layers = []
	print(hr_image_bilinear.get_shape())
	conv = slim.conv2d(hr_image_bilinear, cfg.ngf, [3,3], stride = 2, scope = 'encoder0')
	layers.append(conv)

	layers_specs = [
		cfg.ngf*2, 
		cfg.ngf*4,
		cfg.ngf*8,
		cfg.ngf*8,
		cfg.ngf*8,
		cfg.ngf*8,
	]
	for idx, out_channels in enumerate(layers_specs):
		with slim.arg_scope([slim.conv2d], activation_fn = lrelu, stride = 2, padding = 'VALID'):
			conv = conv2d(layers[-1], out_channels, scope = 'encoder%d'%(idx+1))
			print(conv.get_shape())
			layers.append(conv)
	### decoder part

	layers_specs = [
		(cfg.ngf*8, 0.5),
		(cfg.ngf*8, 0.5),
		(cfg.ngf*8, 0.0),
		(cfg.ngf*4, 0.0),
		(cfg.ngf*2, 0.0),
		(cfg.ngf, 0.0)
	]
	num_encoder_layers = len(layers)

	for decoder_layer_idx, (out_channels, dropout) in enumerate(layers_specs):
		skip_layer = num_encoder_layers - decoder_layer_idx - 1
		with slim.arg_scope([slim.conv2d], activation_fn = lrelu):
			if decoder_layer_idx == 0:
				input = layers[-1]
			else:
				input = tf.concat([layers[-1], layers[skip_layer]], axis = 3)
			output = upsample_layer(input, out_channels, mode = 'deconv')
			print(output.get_shape())
			if dropout > 0.0:
				output = tf.nn.dropout(output, keep_prob = 1 - dropout)
			layers.append(output)
	input = tf.concat([layers[-1],layers[0]], axis = 3)
	output = slim.conv2d_transpose(input, num_channels, [4,4], stride = 2, activation_fn = tf.tanh)
	return output

def create_discriminator(hr_images_fake, hr_images, cfg):
	n_layers = 3
	layers = []

	input = tf.concat([hr_images_fake, hr_images ], axis = 3)

	conv = slim.conv2d(input, cfg.ndf, [3,3], stride = 2, activation_fn = lrelu, scope = 'layers%d'%(0))
	layers.append(conv)

	for i in range(n_layers):
		out_channels = cfg.ndf*min(2**(i+1), 8)
		stride = 1 if i == n_layers -1 else 2
		conv = slim.conv2d(layers[-1], out_channels, [3,3], stride = stride, activation_fn = lrelu, scope = 'layers_%d'%(i+2))
		layers.append(conv)

	conv = slim.conv2d(layers[-1], 1, [3,3], stride = 1)
	output = tf.sigmoid(conv)
	return output
class pix2pix_model():

	def __init__(self, cfg):
		self.cfg = cfg
		#self.constrcut_model(lr_image, hr_image)

	def constrcut_model(self, lr_image, hr_image, phase = 'train'):
		self.lr_image = lr_image
		self.hr_image = hr_image
		self.bilinear_hr_image = tf.image.resize_images(lr_image, shape2size(hr_image))
		with tf.name_scope('generator'):
			with tf.variable_scope("generator"):
				self.fake_hr_image = create_generator(self.bilinear_hr_image, 3, self.cfg)
		with tf.name_scope('discriminator_true'):
			with tf.variable_scope("discriminator"):
				self.fake_score = create_discriminator(self.bilinear_hr_image,self.fake_hr_image, self.cfg)
		with tf.name_scope('discriminator_fake'):
			with tf.variable_scope("discriminator", reuse = True):
				self.true_score = create_discriminator(self.bilinear_hr_image, self.hr_image, self.cfg)

		with tf.name_scope("discriminator_loss"):
			self.d_loss = tf.reduce_mean(-(tf.log(self.true_score + eps) + tf.log(1-self.fake_score+eps)))
		with tf.name_scope("generator_losss"):
			self.g_gan_loss = tf.reduce_mean(-tf.log(self.fake_score+eps))
			self.g_l1_loss = tf.reduce_mean(tf.abs(self.hr_image - self.fake_hr_image))
			self.g_loss = self.g_gan_loss + self.g_l1_loss
		if phase == 'train':

			with tf.name_scope("discriminator_train"):
				d_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
				d_opt = tf.train.AdamOptimizer(self.cfg.lr, self.cfg.beta1)
				d_grads_vars = d_opt.compute_gradients(self.d_loss, var_list = d_vars)
				self.d_train = d_opt.apply_gradients(d_grads_vars)
			with tf.name_scope("generator_loss"):
				with tf.control_dependencies([self.d_train]):
					g_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
					g_opt = tf.train.AdamOptimizer(self.cfg.lr, self.cfg.beta1)
					print(g_vars)
					g_grads_vars = g_opt.compute_gradients(self.g_loss, var_list = g_vars)
					self.g_train = g_opt.apply_gradients(g_grads_vars)
	def test_model(self, lr_image, hr_image = None):
		self.lr_image = lr_image
		self.hr_image = hr_image
		self.bilinear_hr_image = tf.image.resize_images(lr_image, shape2size(hr_image))
		with tf.name_scope('generator'):
			with tf.variable_scope("generator"):
				self.fake_hr_image = create_generator(self.bilinear_hr_image, 3, self.cfg)
def shape2size(tensor):
	shape = tensor.get_shape().as_list()
	h,w = shape[1], shape[2]
	return (h,w)
if __name__ == '__main__':
	class cfg():
		def __init__(self):
			self.ngf = 20
			self.ndf = 20
			self.lr = 0.01
			self.beta1 = 0.9
	cfg = cfg()
	input_lr = tf.ones((20, 32, 32, 3))
	input_hr = tf.ones((20,128,128,3))

	model = pix2pix_model(input_lr, input_hr, cfg)