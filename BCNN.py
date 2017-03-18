
import tensorflow as tf
import numpy as np

class BCNN:
    def __init__(self, s, w, lr, l2_reg, d0=300, di=50):
        """
        Implmenentaion of BCNN

        :param s: sentence length
        :param w: filter width
        :param lr: learning rate
        :param l2_reg: L2 regularization coefficient
        :param d0: dimensionality of word embedding(default: 300)
        :param di: The number of convolution kernels (default: 50)

        """

        self.x1 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x1")
        self.x2 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x2")

        # zero padding to inputs for wide convolution
        self.x1_padded = tf.pad(self.x1, np.array([[0, 0], [0, 0], [w-1, w-1]]), "CONSTANT")
        self.x2_padded = tf.pad(self.x2, np.array([[0, 0], [0, 0], [w-1, w-1]]), "CONSTANT")

        self.y = tf.placehoder(tf.float32, shape=[None], name="y")

        with tf.name_scope("CNN-1 layer"):
            W = tf.Varaible(tf.truncated_normal([d0, w, 1, di], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[di]), name="b")

            conv1 = tf.nn.conv2d(
                # [batch, input_height, input_width, in_channels]
                input = tf.expand_dims(self.x1_padded, -1),
                # [filter_height, filter_width, in_channels, out_channels]
                filter = W,
                stride = [1, 1, 1, 1],
                padding = "VALID",
                name="conv1"
            )
            #output: [batch, 1, input_width - filter_Width -1, out_channels]
            # == [batch, 1, s+w-1, di]

            # [batch, di, s+w-1, 1]
            conv1_reshaped = tf.reshape(conv1, [-1, di, s+w-1, 1])

            activation = tf.nn.tanh(tf.nn.bias_add(conv1_reshaped, b), name="activation")

            pooling = tf.layers.average_pooling2d(
                inputs = activation,
                # (pool_height, pool_width)
                pool_size = (1, w),
                strides = 1,
                padding = "VALID",
                name = "avg_pool"
            )
            # output: [batch, di, s, 1] ??? expected









