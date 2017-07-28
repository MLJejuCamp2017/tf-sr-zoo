import tensorflow as tf

tf.app.flags.DEFINE_integer('batch_num_per_epoch', 1000,
                            """Number of batchs per epoch.""")
tf.app.flags.DEFINE_integer('num_images_epoch', 100000, """
                            Numers of images per epoch
                            """)
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of image patchs to process in a batch.""")
tf.app.flags.DEFINE_integer('patch_size', 128,
                            """The size of image patch.""")
tf.app.flags.DEFINE_integer('num_patch', 64000,
                            """Number  of image patch per epoch..""")
tf.app.flags.DEFINE_integer('num_epochs', 100,
                           """ Epoch to run to genrate the tfrecord""")
tf.app.flags.DEFINE_string('tfrecord_dir', '../dataset/tfrecord/train.record',
                           """ The path of generated tfrecord""")
tf.app.flags.DEFINE_string('datasets_dir', 'datasets',
                           """ The path of dataset""")
tf.app.flags.DEFINE_string('datasets', 'dataset',
                           """ The names of dataset used""")
tf.app.flags.DEFINE_string('mode', 'train', """mode means train or test""")
tf.app.flags.DEFINE_string('log_dir', 'logs', """tensorboard logs dir""")
tf.app.flags.DEFINE_string('train_dir', 'checkpoint', """checkpoint dir""")
tf.app.flags.DEFINE_integer('level', 2, "The laplace level")
tf.app.flags.DEFINE_float('init_lr',0.001,
                           """ The init lr""")
tf.app.flags.DEFINE_float('lr_decay_factor',0.96,
                           """ The lr decay factor""")

if __name__ == '__main__':
    tf.app.run()



