import tensorflow as tf
import numpy as np


class BCNN:
    def __init__(self, s, w, l2_reg, d0=300, di=50, num_classes=2, num_features=1):
        """
        Implmenentaion of BCNN

        :param s: sentence length
        :param w: filter width
        :param lr: learning rate
        :param l2_reg: L2 regularization coefficient
        :param d0: dimensionality of word embedding(default: 300)
        :param di: The number of convolution kernels (default: 50)
        :param num_classes: The number of classes for answers.
        :param num_features: The number of pre-set features(not coming from CNN) used in the output layer.
        """

        self.x1 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x1")
        self.x2 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x2")

        self.features = tf.placeholder(tf.float32, shape=[None, num_features], name="features")

        self.y = tf.placeholder(tf.float32, shape=[None, num_classes], name="y")

        # [filter_height, filter_width, in_channels, out_channels]
        W1 = tf.Variable(tf.truncated_normal([d0, w, 1, di], stddev=0.1), name="W")
        b1 = tf.Variable(tf.constant(0.1, shape=[di]), name="b")

        W2 = tf.Variable(tf.truncated_normal([di, w, 1, di], stddev=0.1), name="W")
        b2 = tf.Variable(tf.constant(0.1, shape=[di]), name="b")

        def CNN(name_scope, x, W, b):
            with tf.name_scope(name_scope):
                conv = tf.nn.conv2d(
                    # [batch, input_height, input_width, in_channels]
                    input=x,
                    # [filter_height, filter_width, in_channels, out_channels]
                    filter=W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )
                # output: [batch, 1, input_width - filter_Width -1, out_channels] == [batch, 1, s+w-1, di]

                activation = tf.nn.tanh(tf.nn.bias_add(conv, b), name="activation")

                # [batch, di, s+w-1, 1]
                reshaped = tf.reshape(activation, [-1, di, s + w - 1, 1], name="reshaped")

                w_ap = tf.layers.average_pooling2d(
                    inputs=reshaped,
                    # (pool_height, pool_width)
                    pool_size=(1, w),
                    strides=1,
                    padding="VALID",
                    name="w_ap"
                )
                # [batch, di, s, 1]

                all_ap = tf.layers.average_pooling2d(
                    inputs=reshaped,
                    # (pool_height, pool_width)
                    pool_size=(1, s + w - 1),
                    strides=1,
                    padding="VALID",
                    name="all_ap"
                )

                return w_ap, tf.reshape(all_ap, [-1, di])

        # zero padding to inputs for wide convolution
        def pad_for_wide_conv(x):
            return tf.pad(x, np.array([[0, 0], [0, 0], [w - 1, w - 1], [0, 0]]), "CONSTANT")

        cnn_1_left_to_cnn_2_left, cnn_1_left_output = CNN(name_scope="CNN-1-left-layer",
                                                          x=pad_for_wide_conv(tf.expand_dims(self.x1,-1)),
                                                          W=W1,
                                                          b=b1)
        self.test = cnn_1_left_to_cnn_2_left

        cnn_1_right_to_cnn_2_right, cnn_1_right_output = CNN(name_scope="CNN-1-right-layer",
                                                             x=pad_for_wide_conv(tf.expand_dims(self.x2,-1)),
                                                             W=W1,
                                                             b=b1)

        _, cnn_2_left_output = CNN(name_scope="CNN-2-left-layer",
                                   x=pad_for_wide_conv(cnn_1_left_to_cnn_2_left),
                                   W=W2,
                                   b=b2)

        _, cnn_2_right_output = CNN(name_scope="CNN-2-right-layer",
                                    x=pad_for_wide_conv(cnn_1_right_to_cnn_2_right),
                                    W=W2,
                                    b=b2)

        """
        with tf.name_scope("CNN-1-left-layer"):
            conv = tf.nn.conv2d(
                # [batch, input_height, input_width, in_channels]
                input=tf.expand_dims(x1_padded, -1),
                # [filter_height, filter_width, in_channels, out_channels]
                filter=W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv"
            )
            # output: [batch, 1, input_width - filter_Width -1, out_channels] == [batch, 1, s+w-1, di]

            activation = tf.nn.tanh(tf.nn.bias_add(conv, b), name="activation")

            # [batch, di, s+w-1, 1]
            reshaped = tf.reshape(activation, [-1, di, s + w - 1, 1], name="reshaped")

            w_ap = tf.layers.average_pooling2d(
                inputs=reshaped,
                # (pool_height, pool_width)
                pool_size=(1, w),
                strides=1,
                padding="VALID",
                name="all_ap"
            )
            # [batch, di, s, 1]

            all_ap = tf.layers.average_pooling2d(
                inputs=reshaped,
                # (pool_height, pool_width)
                pool_size=(1, s + w - 1),
                strides=1,
                padding="VALID",
                name="all_ap"
            )

            cnn_1_left_output = tf.reshape(all_ap, [-1, di])

        with tf.name_scope("CNN-1-right-layer"):
            conv = tf.nn.conv2d(
                input=tf.expand_dims(x2_padded, -1),
                filter=W1,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv"
            )

            activation = tf.nn.tanh(tf.nn.bias_add(conv, b1), name="activation")
            reshaped = tf.reshape(activation, [-1, di, s + w - 1, 1], name="reshaped")

            all_ap = tf.layers.average_pooling2d(
                inputs=reshaped,
                # pool_size = (1, w),
                pool_size=(1, s + w - 1),
                strides=1,
                padding="VALID",
                name="all_ap"
            )

            cnn_1_right_output = tf.reshape(all_ap, [-1, di])
        """

        def consine_similarity(v1, v2):
            norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
            norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
            dot_products = tf.reduce_sum(v1 * v2, axis=1)

            return dot_products / (norm1 * norm2)

        def euclidean_distance_measure(v1, v2):
            return 1 / (1 + tf.sqrt(tf.reduce_sum(tf.square(cnn_1_left_output - cnn_1_right_output), axis=1)))

        with tf.name_scope("output-layer"):
            cosine_sim1 = consine_similarity(cnn_1_left_output, cnn_1_right_output)
            cosine_sim2 = consine_similarity(cnn_2_left_output, cnn_2_right_output)

            euclidean1 = euclidean_distance_measure(cnn_1_left_output, cnn_1_right_output)
            euclidean2 = euclidean_distance_measure(cnn_2_left_output, cnn_2_right_output)

            self.output_features = tf.concat([self.features,
                                              tf.stack([cosine_sim1, cosine_sim2, euclidean1, euclidean2], axis=1)],
                                             axis=1,
                                             name="output_features")

            # self.output_features = tf.concat([cnn_1_left_output, cnn_1_right_output], axis=1)

            output_W = tf.Variable(tf.truncated_normal([5, num_classes], stddev=0.1), name="output_W")
            output_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="output_b")

            estimation = tf.matmul(self.output_features, output_W) + output_b
            prediction = tf.argmax(estimation, axis=1)

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=estimation, labels=self.y)
                                       + l2_reg * tf.nn.l2_loss(W1)
                                       + l2_reg * tf.nn.l2_loss(W2)
                                       + l2_reg * tf.nn.l2_loss(output_W))

            self.accuracy_sum = tf.reduce_sum(
                tf.cast(tf.equal(prediction, tf.argmax(self.y, axis=1)), tf.float32))
            self.accuracy_batch = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(self.y, axis=1)), tf.float32))
