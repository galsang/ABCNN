import tensorflow as tf
import sys
import numpy as np

from preprocess import Word2Vec, MSRP, WikiQA
from ABCNN import ABCNN
from utils import build_path
from sklearn.externals import joblib


def test(w, l2_reg, epoch, max_len, model_type, num_layers, data_type, classifier, word2vec, num_classes=2):
    if data_type == "WikiQA":
        test_data = WikiQA(word2vec=word2vec, max_len=max_len)
    else:
        test_data = MSRP(word2vec=word2vec, max_len=max_len)

    test_data.open_file(mode="test")

    model = ABCNN(s=max_len, w=w, l2_reg=l2_reg, model_type=model_type,
                  num_features=test_data.num_features, num_classes=num_classes, num_layers=num_layers)

    model_path = build_path("./models/", data_type, model_type, num_layers)
    MAPs, MRRs = [], []

    print("=" * 50)
    print("test data size:", test_data.data_size)

    # Due to GTX 970 memory issues
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    for e in range(1, epoch + 1):
        test_data.reset_index()

        #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model_path + "-" + str(e))
            print(model_path + "-" + str(e), "restored.")

            if classifier == "LR" or classifier == "SVM":
                clf_path = build_path("./models/", data_type, model_type, num_layers,
                                      "-" + str(e) + "-" + classifier + ".pkl")
                clf = joblib.load(clf_path)
                print(clf_path, "restored.")

            QA_pairs = {}
            s1s, s2s, labels, features = test_data.next_batch(batch_size=test_data.data_size)

            for i in range(test_data.data_size):
                pred, clf_input = sess.run([model.prediction, model.output_features],
                                           feed_dict={model.x1: np.expand_dims(s1s[i], axis=0),
                                                      model.x2: np.expand_dims(s2s[i], axis=0),
                                                      model.y: np.expand_dims(labels[i], axis=0),
                                                      model.features: np.expand_dims(features[i], axis=0)})

                if classifier == "LR":
                    clf_pred = clf.predict_proba(clf_input)[:, 1]
                    pred = clf_pred
                elif classifier == "SVM":
                    clf_pred = clf.decision_function(clf_input)
                    pred = clf_pred

                s1 = " ".join(test_data.s1s[i])
                s2 = " ".join(test_data.s2s[i])

                if s1 in QA_pairs:
                    QA_pairs[s1].append((s2, labels[i], np.asscalar(pred)))
                else:
                    QA_pairs[s1] = [(s2, labels[i], np.asscalar(pred))]

            # Calculate MAP and MRR for comparing performance
            MAP, MRR = 0, 0
            for s1 in QA_pairs.keys():
                p, AP = 0, 0
                MRR_check = False

                QA_pairs[s1] = sorted(QA_pairs[s1], key=lambda x: x[-1], reverse=True)

                for idx, (s2, label, prob) in enumerate(QA_pairs[s1]):
                    if label == 1:
                        if not MRR_check:
                            MRR += 1 / (idx + 1)
                            MRR_check = True

                        p += 1
                        AP += p / (idx + 1)

                AP /= p
                MAP += AP

            num_questions = len(QA_pairs.keys())
            MAP /= num_questions
            MRR /= num_questions

            MAPs.append(MAP)
            MRRs.append(MRR)

            print("[Epoch " + str(e) + "] MAP:", MAP, "/ MRR:", MRR)

    print("=" * 50)
    print("max MAP:", max(MAPs), "max MRR:", max(MRRs))
    print("=" * 50)

    exp_path = build_path("./experiments/", data_type, model_type, num_layers, "-" + classifier + ".txt")
    with open(exp_path, "w", encoding="utf-8") as f:
        print("Epoch\tMAP\tMRR", file=f)
        for i in range(e):
            print(str(i + 1) + "\t" + str(MAPs[i]) + "\t" + str(MRRs[i]), file=f)


if __name__ == "__main__":

    # Paramters
    # --ws: window_size
    # --l2_reg: l2_reg modifier
    # --epoch: epoch
    # --max_len: max sentence length
    # --model_type: model type
    # --num_layers: number of convolution layers
    # --data_type: MSRP or WikiQA data
    # --classifier: Final layout classifier(model, LR, SVM)

    # default parameters
    params = {
        "ws": 4,
        "l2_reg": 0.0004,
        "epoch": 50,
        "max_len": 40,
        "model_type": "BCNN",
        "num_layers": 2,
        "data_type": "WikiQA",
        "classifier": "LR",
        "word2vec": Word2Vec()
    }

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            k = arg.split("=")[0][2:]
            v = arg.split("=")[1]
            params[k] = v

    test(w=int(params["ws"]), l2_reg=float(params["l2_reg"]), epoch=int(params["epoch"]),
         max_len=int(params["max_len"]), model_type=params["model_type"],
         num_layers=int(params["num_layers"]), data_type=params["data_type"],
         classifier=params["classifier"], word2vec=params["word2vec"])
