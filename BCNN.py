
import tensorflow as tf

class BCNN:
    def __init__(self, d0=300, s):

        x1 = tf.placeholder(tf.float32, [None, d0, s], name="X1")
        x2 = tf.placeholder(tf.float32, [None, d0, s], name="X2")

        with tf.name_scope("CNN"):
            self.CNN = tf.contrib.layers.conv2d(
                [None, d0, s, 1]
                weights_initializer =
                weights_regularizer =
                biases_initializer =

            )







