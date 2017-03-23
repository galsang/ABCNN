import tensorflow as tf
import numpy as np


class ABCNN():
    def __init__(self, s, w, l2_reg, batch_size, d0=300, di=50, num_classes=2, num_features=1, type="BCNN"):
        """
        Implmenentaion of 2-layer ABCNNs
        (https://arxiv.org/pdf/1512.05193.pdf)

        :param s: sentence length
        :param w: filter width
        :param lr: learning rate
        :param l2_reg: L2 regularization coefficient
        :param batch_size: The size of each batch of data.
        :param d0: dimensionality of word embedding(default: 300)
        :param di: The number of convolution kernels (default: 50)
        :param num_classes: The number of classes for answers.
        :param num_features: The number of pre-set features(not coming from CNN) used in the output layer.
        """

        self.x1 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x1")
        self.x2 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x2")

        self.features = tf.placeholder(tf.float32, shape=[None, num_features], name="features")

        self.y = tf.placeholder(tf.float32, shape=[None, num_classes], name="y")

        self.L2_term = 0

        if type == "ABCNN1" or type == "ABCNN3":
            in_channels = 2
        else:
            in_channels = 1

        # zero padding to inputs for wide convolution
        def pad_for_wide_conv(x):
            return tf.pad(x, np.array([[0, 0], [0, 0], [w - 1, w - 1], [0, 0]]), "CONSTANT")

        def cos_sim(v1, v2):
            norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
            norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
            dot_products = tf.reduce_sum(v1 * v2, axis=1)

            return dot_products / (norm1 * norm2)

        def euclidean_score(v1, v2):
            return 1 / (1 + tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1)))

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

        def CNN_layer(name_scope, x1, x2, d):
            # x1, x2 = [batch, d, s, 1]
            with tf.name_scope(name_scope):
                if type == "ABCNN1":
                    attention_W = tf.Variable(tf.truncated_normal([s, d], stddev=0.1), name="aW")

                    # [batch, s, s]
                    A = tf.sqrt(tf.reduce_sum(tf.sqaure(x1 - tf.reshape(x2, [-1, d, 1, s])), axis=1))
                    self.test = tf.shape(A)

                    # [batch, d, s, 1]
                    x1_a = tf.expand_dims(tf.matrix_transpose(tf.matmul(A, attention_W)), -1)
                    x2_a = tf.expand_dims(tf.matrix_transpose(tf.matmul(tf.matrix_transpose(A), attention_W)), -1)

                    # [batch, d, s, 2]
                    x1 = tf.concat([x1, x1_a], axis=3)
                    x2 = tf.concat([x2, x2_a], axis=3)

                # [filter_height, filter_width, in_channels, out_channels]
                conv_W = tf.Variable(tf.truncated_normal([d, w, in_channels, di], stddev=0.1), name="conv_W")
                conv_b = tf.Variable(tf.constant(0.1, shape=[di]), name="conv_b")

                self.L2_term += l2_reg * tf.nn.l2_loss(conv_W)

                left_w_ap, left_all_ap = CNN(name_scope="left",
                                             x=pad_for_wide_conv(x1),
                                             W=conv_W,
                                             b=conv_b)

                right_w_ap, right_all_ap = CNN(name_scope="right",
                                               x=pad_for_wide_conv(x2),
                                               W=conv_W,
                                               b=conv_b)

            return left_w_ap, left_all_ap, right_w_ap, right_all_ap

        cnn_1_left_to_cnn_2_left, cnn_1_left_output, cnn_1_right_to_cnn_2_right, cnn_1_right_output = \
            CNN_layer(name_scope="CNN-1",
                      x1=tf.expand_dims(self.x1, -1),
                      x2=tf.expand_dims(self.x2, -1),
                      d=d0)

        _, cnn_2_left_output, _, cnn_2_right_output = CNN_layer(name_scope="CNN-2",
                                                                x1=cnn_1_left_to_cnn_2_left,
                                                                x2=cnn_1_right_to_cnn_2_right,
                                                                d=di)

        with tf.name_scope("output-layer"):
            cosine_sim1 = cos_sim(cnn_1_left_output, cnn_1_right_output)
            cosine_sim2 = cos_sim(cnn_2_left_output, cnn_2_right_output)

            euclidean1 = euclidean_score(cnn_1_left_output, cnn_1_right_output)
            euclidean2 = euclidean_score(cnn_2_left_output, cnn_2_right_output)

            self.output_features = tf.concat([self.features,
                                              tf.stack([cosine_sim1, cosine_sim2, euclidean1, euclidean2], axis=1)],
                                             axis=1,
                                             name="output_features")

            output_W = tf.Variable(tf.truncated_normal([5, num_classes], stddev=0.1), name="output_W")
            output_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="output_b")

            self.L2_term += l2_reg * tf.nn.l2_loss(output_W)

            estimation = tf.matmul(self.output_features, output_W) + output_b
            prediction = tf.argmax(estimation, axis=1)

            self.L2_term += l2_reg * tf.nn.l2_loss(output_W)

            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=estimation, labels=self.y) + self.L2_term)

            self.accuracy_sum = tf.reduce_sum(
                tf.cast(tf.equal(prediction, tf.argmax(self.y, axis=1)), tf.float32))
            self.accuracy_batch = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(self.y, axis=1)), tf.float32))
