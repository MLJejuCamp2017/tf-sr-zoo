import tensorflow as tf
tf.app.flags.DEFINE_integer('num_batches_per_epoch', 3000,
                            """Number of batchs per epoch.""")
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of image patchs to process in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', 20, """num of epoch to run""")
tf.app.flags.DEFINE_string('tfrecord_dir', '../dataset/tfrecord/train.record',
                           """ The path of generated tfrecord""")
tf.app.flags.DEFINE_string('log_dir', 'logs', """tensorboard logs dir""")
tf.app.flags.DEFINE_string('train_dir', 'checkpoint', """checkpoint dir""")
tf.app.flags.DEFINE_float('lr', 0.0001, """init lr""")
tf.app.flags.DEFINE_float('beta1', 0.9, """beta for adam""")
tf.app.flags.DEFINE_integer('ngf', 32, """ngf""")
tf.app.flags.DEFINE_integer('ndf', 32,"""ndf""" )
cfg = tf.app.flags.FLAGS
