import tensorflow as tf
import numpy as np


class ABCNN():
    def __init__(self, s, w, l2_reg, model_type, d0=300, di=50, num_classes=2, num_features=1):
        """
        Implmenentaion of 2-layer ABCNNs
        (https://arxiv.org/pdf/1512.05193.pdf)

        :param s: sentence length
        :param w: filter width
        :param l2_reg: L2 regularization coefficient
        :param model_type: Type of the network(BCNN, ABCNN1, ABCNN2, ABCNN3).
        :param d0: dimensionality of word embedding(default: 300)
        :param di: The number of convolution kernels (default: 50)
        :param num_classes: The number of classes for answers.
        :param num_features: The number of pre-set features(not coming from CNN) used in the output layer.
        """

        self.x1 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x1")
        self.x2 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x2")

        self.features = tf.placeholder(tf.float32, shape=[None, num_features], name="features")

        self.y = tf.placeholder(tf.float32, shape=[None, num_classes], name="y")

        L2_term = tf.constant(0.0)

        if model_type == "ABCNN1" or model_type == "ABCNN3":
            in_channels = 2
        else:
            in_channels = 1

        # zero padding to inputs for wide convolution
        def pad_for_wide_conv(x):
            return tf.pad(x, np.array([[0, 0], [0, 0], [w - 1, w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

        def cos_sim(v1, v2):
            norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
            norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
            dot_products = tf.reduce_sum(v1 * v2, axis=1)

            return dot_products / (norm1 * norm2)

        def euclidean_score(v1, v2):
            return 1 / (1 + tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1)))

        def make_attention_mat(x1, x2, height, width):
            # x1, x2 = [batch, height, width, 1]
            return tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.reshape(x2, [-1, height, 1, width])), axis=1),
                           name="attention_mat")

        def convolution(name_scope, x, W, b):
            with tf.name_scope(name_scope + "-conv"):
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
                return tf.reshape(activation, [-1, di, s + w - 1, 1], name="conv_result")

        def pooling(name_scope, x, attention):
            # x: [batch, di, s+w-1, 1]
            # attention: [batch, s+w-1]
            with tf.name_scope(name_scope + "-pooling"):
                if model_type == "ABCNN2" or model_type == "ABCNN3":
                    pools = []
                    # [batch, 1, s+w-1, 1]
                    attention = tf.reshape(attention, [-1, 1, s + w - 1, 1])

                    for i in range(s):
                        # [batch, di, 1, 1]
                        pools.append(tf.reduce_sum(x[:, :, i:i + w, :] * attention[:, :, i:i + w, :],
                                                   axis=2,
                                                   keep_dims=True))

                    # [batch, di, s, 1]
                    w_p = tf.concat(pools, axis=2, name="w_p")
                    # [batch, di, 1, 1]
                    all_p = tf.reduce_sum(x * attention, axis=2, name="all_p")

                    return w_p, tf.reshape(all_p, [-1, di]),
                else:
                    w_ap = tf.layers.average_pooling2d(
                        inputs=x,
                        # (pool_height, pool_width)
                        pool_size=(1, w),
                        strides=1,
                        padding="VALID",
                        name="w_ap"
                    )
                    # [batch, di, s, 1]

                    all_ap = tf.layers.average_pooling2d(
                        inputs=x,
                        # (pool_height, pool_width)
                        pool_size=(1, s + w - 1),
                        strides=1,
                        padding="VALID",
                        name="all_ap"
                    )

                    return w_ap, tf.reshape(all_ap, [-1, di])

        def CNN_layer(name_scope, x1, x2, d):
            L2_term_part = tf.constant(0.0)

            # x1, x2 = [batch, d, s, 1]
            with tf.name_scope(name_scope):
                if model_type == "ABCNN1" or model_type == "ABCNN3":
                    attention_W = tf.Variable(tf.truncated_normal([s, d], stddev=0.1), name="aW")
                    L2_term_part += tf.nn.l2_loss(attention_W)

                    # [batch, s, s]
                    A = make_attention_mat(x1, x2, d, s)

                    # [batch, d, s, 1]
                    x1_a = tf.expand_dims(tf.matrix_transpose(
                        tf.einsum("ijk,kl->ijl", A, attention_W)), -1)
                    x2_a = tf.expand_dims(tf.matrix_transpose(
                        tf.einsum("ijk,kl->ijl", tf.matrix_transpose(A), attention_W)), -1)

                    # [batch, d, s, 2]
                    x1 = tf.concat([x1, x1_a], axis=3)
                    x2 = tf.concat([x2, x2_a], axis=3)

                # [filter_height, filter_width, in_channels, out_channels]
                conv_W = tf.Variable(tf.truncated_normal([d, w, in_channels, di], stddev=0.1), name="conv_W")
                L2_term_part += tf.nn.l2_loss(conv_W)

                conv_b = tf.Variable(tf.constant(0.01, shape=[di]), name="conv_b")

                left_conv = convolution(name_scope="left",
                                        x=pad_for_wide_conv(x1),
                                        W=conv_W,
                                        b=conv_b)

                right_conv = convolution(name_scope="right",
                                         x=pad_for_wide_conv(x2),
                                         W=conv_W,
                                         b=conv_b)

                left_attention, right_attention = None, None

                if model_type == "ABCNN2" or model_type == "ABCNN3":
                    A = make_attention_mat(left_conv, right_conv, di, s + w - 1)

                    left_attention = tf.reduce_sum(A, axis=1)
                    right_attention = tf.reduce_sum(A, axis=2)

                left_wp, left_ap = pooling(name_scope="left",
                                           x=left_conv,
                                           attention=left_attention)

                right_wp, right_ap = pooling(name_scope="right",
                                             x=right_conv,
                                             attention=right_attention)

            return left_wp, left_ap, right_wp, right_ap, L2_term_part

        LI_1, LO_1, RI_1, RO_1, L2_term_1 = CNN_layer(name_scope="CNN-1",
                                                      x1=tf.expand_dims(self.x1, -1),
                                                      x2=tf.expand_dims(self.x2, -1),
                                                      d=d0)

        _, LO_2, _, RO_2, L2_term_2 = CNN_layer(name_scope="CNN-2", x1=LI_1, x2=RI_1, d=di)

        L2_term += L2_term_1 + L2_term_2

        with tf.name_scope("output-layer"):
            cosine_sim1 = cos_sim(LO_1, RO_1)
            cosine_sim2 = cos_sim(LO_2, RO_2)

            euclidean1 = euclidean_score(LO_1, RO_1)
            euclidean2 = euclidean_score(LO_2, RO_2)

            """
            self.output_features = tf.stack([cosine_sim1, cosine_sim2, euclidean1, euclidean2], axis=1,
                                            name="output_features")
            """

            """
            self.output_features = tf.concat([self.features,
                                              tf.stack([euclidean1, euclidean2], axis=1)],
                                             axis=1,
                                             name="output_features")
            """
            self.output_features= tf.concat([LO_1, RO_1, LO_2, RO_2], axis=1, name="output_features")

            output_W = tf.Variable(tf.truncated_normal([di * 4, num_classes], stddev=0.1), name="output_W")
            L2_term += tf.nn.l2_loss(output_W)

            output_b = tf.Variable(tf.constant(0.01, shape=[num_classes]), name="output_b")

            estimation = tf.matmul(self.output_features, output_W) + output_b
            prediction = tf.argmax(estimation, axis=1)

            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=estimation, labels=self.y)) + l2_reg * L2_term
            tf.summary.scalar("cost", self.cost)

            self.batch_accuracy_sum = tf.reduce_sum(
                tf.cast(tf.equal(prediction, tf.argmax(self.y, axis=1)), tf.float32))

            self.batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(self.y, axis=1)), tf.float32))
            tf.summary.scalar("batch_accuracy", self.batch_accuracy)

            self.merged = tf.summary.merge_all()
