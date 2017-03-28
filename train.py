# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from preprocess import Word2Vec, MSRP
from ABCNN import ABCNN


def train(lr, w, l2_reg, batch_size, model_type, word2vec, num_classes=2):
    train_data = MSRP(mode="train", word2vec=word2vec)
    dev_data = MSRP(mode="dev", word2vec=word2vec, max_len=train_data.max_len)

    print("=" * 50)
    print("training data size:", train_data.data_size)
    print("dev data size:", dev_data.data_size)

    model = ABCNN(s=train_data.max_len, w=w, l2_reg=l2_reg, num_classes=num_classes,
                  num_features=train_data.num_features, model_type=model_type)

    optimizer = tf.train.AdagradOptimizer(lr).minimize(model.cost)

    # Due to GTX 970 memory issues
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    # Initialize all variables
    init = tf.global_variables_initializer()

    # model(parameters) saver
    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_summary_writer = tf.summary.FileWriter("C:/tf_logs/train", sess.graph)
        # dev_summary_writer = tf.summary.FileWriter("C:/tf_logs/dev", sess.graph)

        sess.run(init)

        i = 0
        while train_data.is_available():
            batch_x1, batch_x2, batch_y, batch_features = sess.run(train_data.next_batch(batch_size=batch_size,
                                                                                         num_classes=num_classes))

            summary, _, c, train_acc = sess.run([model.merged, optimizer, model.cost, model.batch_accuracy],
                                                feed_dict={model.x1: batch_x1,
                                                           model.x2: batch_x2,
                                                           model.y: batch_y,
                                                           model.features: batch_features})

            """
            dev_data.reset_index()
            dev_acc = 0
            while dev_data.is_available():
                dev_x1, dev_x2, dev_y, dev_features = sess.run(dev_data.next_batch(batch_size=400,
                                                                                   num_classes=num_classes))

                dev_summary, acc = sess.run([model.merged, model.batch_accuracy_sum],
                                            feed_dict={model.x1: dev_x1,
                                                       model.x2: dev_x2,
                                                       model.y: dev_y,
                                                       model.features: dev_features})
                dev_acc += acc
            dev_acc = dev_acc / dev_data.data_size
            """

            i += 1
            print("cost " + str(i) + ": ", c, "/ train_batch_acc:", train_acc)  # , "/ dev_acc:", dev_acc)
            train_summary_writer.add_summary(summary, i)
            # dev_summary_writer.add_summary(dev_summary, i)

            save_path = saver.save(sess, "./models/" + model_type, global_step=i)
            if i == 10:
                break

        print("training finished!")
        print("=" * 50)


word2vec = Word2Vec()
train(lr=0.01, w=3, l2_reg=0.0, batch_size=64, model_type="BCNN", word2vec=word2vec)
