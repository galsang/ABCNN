import tensorflow as tf
import sys
import numpy as np

from preprocess import Word2Vec, MSRP, WikiQA
from ABCNN import ABCNN


def test(w, l2_reg, data_type, max_len, model_type, model_path, word2vec, num_classes=2):
    if data_type == "WikiQA":
        test_data = WikiQA(word2vec=word2vec, max_len=max_len)
    else:
        test_data = MSRP(word2vec=word2vec, max_len=max_len)

    test_data.open_file(mode="test")

    model = ABCNN(s=max_len, w=w, l2_reg=l2_reg, model_type=model_type,
                  num_features=test_data.num_features, num_classes=num_classes)

    print("=" * 50)
    print("test data size:", test_data.data_size)

    # Due to GTX 970 memory issues
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print(model_path, "Model restored.")

        QA_pairs = {}
        s1s, s2s, labels, features = test_data.next_batch(batch_size=test_data.data_size)

        for i in range(test_data.data_size):
            pred = sess.run(model.prediction, feed_dict={model.x1: np.expand_dims(s1s[i], axis=0),
                                                         model.x2: np.expand_dims(s2s[i], axis=0),
                                                         model.y: np.expand_dims(labels[i], axis=0),
                                                         model.features: np.expand_dims(features[i], axis=0)})

            s1 = " ".join(test_data.s1s[i])
            s2 = " ".join(test_data.s2s[i])

            if s1 in QA_pairs:
                QA_pairs[s1].append((s2, labels[i], np.asscalar(pred)))
            else:
                QA_pairs[s1] = [(s2, labels[i], np.asscalar(pred))]

        # Calculate MAP and MRR for comparing performance
        MAP, MRR = 0, 0
        for s1 in QA_pairs.keys():
            QA_pairs[s1] = sorted(QA_pairs[s1], key=lambda x: x[-1], reverse=True)

            for idx, (s2, label, prob) in enumerate(QA_pairs[s1]):
                if label == 1:
                    MRR += 1 / (idx + 1)
                    break

        for s1 in QA_pairs.keys():
            p, AP = 0, 0
            for idx, (s2, label, prob) in enumerate(QA_pairs[s1]):
                if label == 1:
                    p += 1
                    AP += p / (idx + 1)

            AP /= p
            MAP += AP

        num_questions = len(QA_pairs.keys())
        MAP /= num_questions
        MRR /= num_questions

        print("MAP:", MAP, "MRR:", MRR)


if __name__ == "__main__":

    # Paramters
    # --ws: window_size
    # --l2_reg: l2_reg modifier
    # --data_type: MSRP or WikiQA data
    # --max_len: max sentence length
    # --model_type: model type
    # --model_path: path of saved model

    # default parameters
    params = {
        "ws": 4,
        "l2_reg": 0.0004,
        "data_type": "WikiQA",
        "max_len": 40,
        "model_type": "ABCNN1",
        "model_path": "./models/WikiQA-ABCNN1-20",
        "word2vec": Word2Vec(),
    }

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            k = arg.split("=")[0][2:]
            v = arg.split("=")[1]
            params[k] = v

    test(w=int(params["ws"]), l2_reg=float(params["l2_reg"]),
         data_type=params["data_type"], max_len=int(params["max_len"]),
         model_type=params["model_type"], model_path=params["model_path"], word2vec=params["word2vec"])
