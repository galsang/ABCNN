import tensorflow as tf

from preprocess import Word2Vec, MSRP
from ABCNN import ABCNN


def test(w, l2_reg, batch_size, model_type, model_path, word2vec, max_len=30, num_classes=2):
    test_data = MSRP(mode="test", word2vec=word2vec, max_len=max_len)

    # print("test data size:", test_data.data_size)

    model = ABCNN(s=max_len, w=w, l2_reg=l2_reg, num_classes=num_classes,
                  num_features=test_data.num_features, model_type=model_type)

    # Due to GTX 970 memory issues
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    # Initialize all variables
    init = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)

        saver = tf.train.import_meta_graph(model_path + ".meta")
        saver.restore(sess, model_path)

        test_acc = 0
        i = 0

        while test_data.is_available():
            batch_x1, batch_x2, batch_y, batch_features = sess.run(test_data.next_batch(batch_size=batch_size,
                                                                                        num_classes=num_classes))

            test_acc += sess.run(model.batch_accuracy_sum, feed_dict={model.x1: batch_x1,
                                                                      model.x2: batch_x2,
                                                                      model.y: batch_y,
                                                                      model.features: batch_features})
            i += 1
            break

        test_acc = test_acc / test_data.data_size  # (batch_size * i)

        print("test_acc:", test_acc)


word2vec = Word2Vec()
for i in range(6, 11):
    test(w=3, l2_reg=0.0, batch_size=256, model_type="BCNN", model_path="./models/BCNN-" + str(i), word2vec=word2vec)
