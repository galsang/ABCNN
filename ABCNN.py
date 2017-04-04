import tensorflow as tf
import numpy as np


class ABCNN():
    def __init__(self, s, w, l2_reg, model_type, num_features, d0=300, di=50, num_classes=2):
        """
        Implmenentaion of 2-layer ABCNNs
        (https://arxiv.org/pdf/1512.05193.pdf)

        :param s: sentence length
        :param w: filter width
        :param l2_reg: L2 regularization coefficient
        :param model_type: Type of the network(BCNN, ABCNN1, ABCNN2, ABCNN3).
        :param num_features: The number of pre-set features(not coming from CNN) used in the output layer.
        :param d0: dimensionality of word embedding(default: 300)
        :param di: The number of convolution kernels (default: 50)
        :param num_classes: The number of classes for answers.
        """

        self.x1 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x1")
        self.x2 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x2")
        self.y = tf.placeholder(tf.int32, shape=[None], name="y")
        self.features = tf.placeholder(tf.float32, shape=[None, num_features], name="features")

        # zero padding to inputs for wide convolution
        def pad_for_wide_conv(x):
            return tf.pad(x, np.array([[0, 0], [0, 0], [w - 1, w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

        def cos_sim(v1, v2):
            norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
            norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
            dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

            return dot_products / (norm1 * norm2)

        def euclidean_score(v1, v2):
            return tf.div(1.0, 1.0 + tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1)), name="eucliean")

        def make_attention_mat(x1, x2, height, width):
            # x1, x2 = [batch, height, width, 1]
            return tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.reshape(x2, [-1, height, 1, width])), axis=1))

        def convolution(name_scope, x, d):
            with tf.name_scope(name_scope + "-conv"):
                with tf.variable_scope("conv") as scope:
                    conv = tf.contrib.layers.conv2d(
                        inputs=x,
                        num_outputs=di,
                        kernel_size=(d, w),
                        stride=1,
                        padding="VALID",
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                        biases_initializer=tf.constant_initializer(1e-04),
                        reuse=True,
                        trainable=True,
                        scope=scope
                    )
                    # Weight: [filter_height, filter_width, in_channels, out_channels]
                    # output: [batch, 1, input_width+filter_Width-1, out_channels] == [batch, 1, s+w-1, di]

                # [batch, di, s+w-1, 1]
                conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
                return conv_trans

        def pooling(variable_scope, x, attention):
            # x: [batch, di, s+w-1, 1]
            # attention: [batch, s+w-1]
            with tf.variable_scope(variable_scope + "-pooling"):
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
                    # [batch, di, 1, 1]

                    # [batch, di]
                    all_ap_reshaped = tf.reshape(all_ap, [-1, di])

                return w_ap, all_ap_reshaped

        def CNN_layer(variable_scope, x1, x2, d):
            # x1, x2 = [batch, d, s, 1]
            with tf.variable_scope(variable_scope):
                if model_type == "ABCNN1" or model_type == "ABCNN3":
                    with tf.name_scope("att_mat"):
                        aW = tf.get_variable(name="aW",
                                             shape=(s, d),
                                             initializer=tf.random_normal_initializer(stddev=1e-02),
                                             regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg))

                        # [batch, s, s]
                        att_mat = make_attention_mat(x1, x2, d, s)

                        # [batch, d, s, 1]
                        x1_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", att_mat, aW)), -1)
                        x2_a = tf.expand_dims(tf.matrix_transpose(
                            tf.einsum("ijk,kl->ijl", tf.matrix_transpose(att_mat), aW)), -1)

                        # [batch, d, s, 2]
                        x1 = tf.concat([x1, x1_a], axis=3)
                        x2 = tf.concat([x2, x2_a], axis=3)

                left_conv = convolution(name_scope="left", x=pad_for_wide_conv(x1), d=d)
                right_conv = convolution(name_scope="right", x=pad_for_wide_conv(x2), d=d)

                left_attention, right_attention = None, None

                if model_type == "ABCNN2" or model_type == "ABCNN3":
                    att_mat = make_attention_mat(left_conv, right_conv, di, s + w - 1)

                    left_attention = tf.reduce_sum(att_mat, axis=1)
                    right_attention = tf.reduce_sum(att_mat, axis=2)

                left_wp, left_ap = pooling(variable_scope="left",
                                           x=left_conv,
                                           attention=left_attention)

                right_wp, right_ap = pooling(variable_scope="right",
                                             x=right_conv,
                                             attention=right_attention)

            return left_wp, left_ap, right_wp, right_ap

        LI_1, LO_1, RI_1, RO_1 = CNN_layer(variable_scope="CNN-1",
                                           x1=tf.expand_dims(self.x1, -1),
                                           x2=tf.expand_dims(self.x2, -1),
                                           d=d0)

        _, LO_2, _, RO_2 = CNN_layer(variable_scope="CNN-2", x1=LI_1, x2=RI_1, d=di)

        with tf.variable_scope("output-layer"):
            cosine_sim1 = cos_sim(LO_1, RO_1)
            cosine_sim2 = cos_sim(LO_2, RO_2)

            self.output_features = tf.concat([self.features,
                                              tf.stack([cosine_sim1, cosine_sim2], axis=1)],
                                             axis=1,
                                             name="output_features")

            self.estimation = tf.contrib.layers.fully_connected(
                inputs=self.output_features,
                num_outputs=num_classes,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(stddev=1e-02),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

        self.prediction = tf.contrib.layers.softmax(self.estimation)[:, 1]

        self.cost = tf.add(
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.estimation, labels=self.y)),
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
            name="cost")

        tf.summary.scalar("cost", self.cost)
        self.merged = tf.summary.merge_all()

        print("=" * 50)
        print("List of Variables:")
        for v in tf.trainable_variables():
            print(v.name)
        print("=" * 50)
