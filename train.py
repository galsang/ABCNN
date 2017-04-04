import tensorflow as tf
import sys

from preprocess import Word2Vec, MSRP, WikiQA
from ABCNN import ABCNN


def train(lr, w, l2_reg, batch_size, model_type, data_type, epoch, word2vec, num_classes=2):
    if data_type == "WikiQA":
        train_data = WikiQA(word2vec=word2vec)
    else:
        train_data = MSRP(word2vec=word2vec)

    train_data.open_file(mode="train")

    print("=" * 50)
    print("training data size:", train_data.data_size)
    print("training max len:", train_data.max_len)
    print("=" * 50)

    model = ABCNN(s=train_data.max_len, w=w, l2_reg=l2_reg, model_type=model_type,
                  num_features=train_data.num_features, num_classes=num_classes)

    optimizer = tf.train.AdagradOptimizer(lr, name="optimizer").minimize(model.cost)

    # Due to GTX 970 memory issues
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    # Initialize all variables
    init = tf.global_variables_initializer()

    # model(parameters) saver
    saver = tf.train.Saver(max_to_keep=20)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_summary_writer = tf.summary.FileWriter("C:/tf_logs/train", sess.graph)

        sess.run(init)

        print("=" * 50)
        for e in range(1, epoch + 1):
            print("[epoch " + str(e) + "]")

            train_data.reset_index()
            i= 0

            while train_data.is_available():
                i += 1

                batch_x1, batch_x2, batch_y, batch_features = train_data.next_batch(batch_size=batch_size)

                merged, _, c = sess.run([model.merged, optimizer, model.cost],
                                        feed_dict={model.x1: batch_x1,
                                                   model.x2: batch_x2,
                                                   model.y: batch_y,
                                                   model.features: batch_features})

                if i % 20 == 0:
                    print("[batch " + str(i) + "] cost:", c)
                train_summary_writer.add_summary(merged, i)

            save_path = saver.save(sess, "./models/" + data_type + "-" + model_type, global_step=e)
            print("model saved in", save_path)

        print("training finished!")
        print("=" * 50)


if __name__ == "__main__":

    # Paramters
    # --lr: learning rate
    # --ws: window_size
    # --l2_reg: l2_reg modifier
    # --epoch: epoch
    # --batch_size: batch size
    # --model_type: model type
    # --data_type: MSRP or WikiQA data

    # default parameters
    params = {
        "lr": 0.08,
        "ws": 4,
        "l2_reg": 0.0004,
        "epoch": 10,
        "batch_size": 64,
        "model_type": "ABCNN1",
        "word2vec": Word2Vec(),
        "data_type": "WikiQA"
    }

    print("=" * 50)
    print("Parameters:")
    for k in params:
        print(k, ":", params[k])

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            k = arg.split("=")[0][2:]
            v = arg.split("=")[1]
            params[k] = v

    train(lr=float(params["lr"]), w=int(params["ws"]), l2_reg=float(params["l2_reg"]), epoch=int(params["epoch"]),
          batch_size=int(params["batch_size"]), model_type=params["model_type"], data_type=params["data_type"],
          word2vec=params["word2vec"])
