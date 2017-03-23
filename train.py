# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from preprocess import MSRP
from ABCNN import ABCNN


def main(lr=0.08, w=3, l2_reg=0.0002, batch_size=400, num_classes=2, type="BCNN"):
    train_data = MSRP(mode="train", batch_size=batch_size, num_classes=num_classes)
    test_data = MSRP(mode="test", batch_size=batch_size, num_classes=num_classes, max_length=train_data.max_length)

    print("=" * 50)
    print("training data size:", train_data.data_size)
    print("test data size:", test_data.data_size)
    print("=" * 50)

    model = ABCNN(s=train_data.max_length,
                  w=w,
                  l2_reg=l2_reg,
                  batch_size=batch_size,
                  num_classes=num_classes,
                  type=type)

    optimizer = tf.train.AdagradOptimizer(lr).minimize(model.cost)

    # Due to GTX 970 memory issues
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # trainig phase
        i = 0
        while train_data.is_available():
            batch_x1, batch_x2, batch_y, batch_features = sess.run(train_data.next_batch())

            print(sess.run(model.test, feed_dict={model.x1: batch_x1,
                                                  model.x2: batch_x2,
                                                  model.features: batch_features,
                                                  model.y: batch_y}))

            _, c, train_acc = sess.run([optimizer, model.cost, model.accuracy_batch],
                                       feed_dict={model.x1: batch_x1,
                                                  model.x2: batch_x2,
                                                  model.features: batch_features,
                                                  model.y: batch_y})

            i += 1
            print("cost " + str(i) + ": ", c, "/ train_batch_acc:", train_acc)

        print("training finished!")
        print("=" * 50)

        # test phase
        test_acc = 0
        while test_data.is_available():
            batch_x1, batch_x2, batch_y, batch_features = sess.run(test_data.next_batch())

            test_acc += sess.run(model.accuracy_sum,
                                 feed_dict={model.x1: batch_x1,
                                            model.x2: batch_x2,
                                            model.features: batch_features,
                                            model.y: batch_y})

        test_acc = test_acc / test_data.data_size
        print("test_acc:", test_acc)
        print("test finished!")
        print("=" * 50)


main(lr=0.001, l2_reg=0.0002, batch_size=100, type="ABCNN1")
