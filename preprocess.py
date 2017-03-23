import tensorflow as tf
import numpy as np
import nltk
import gensim


class Word2Vec():
    def __init__(self):
        # Load Google's pre-trained Word2Vec model.
        self.model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',
                                                                     binary=True)
        self.unknowns = np.random.uniform(-0.01, 0.01, 300).astype("float32")

    def get(self, word):
        if word not in self.model.vocab:
            return self.unknowns
        else:
            return self.model.word_vec(word)


class MSRP():
    def __init__(self, mode="train", batch_size=100, num_classes=2, max_length=0):
        with open("msr_paraphrase_" + mode + ".txt", "r", encoding="utf-8") as f:
            self.data = []
            self.index = -1
            self.max_length = max_length
            self.batch_size = batch_size
            self.num_classes = num_classes
            self.word2vec = Word2Vec()

            f.readline()

            for line in f:
                label = line[:-1].split("\t")[0]
                s1 = nltk.word_tokenize(line[:-1].split("\t")[3])
                s2 = nltk.word_tokenize(line[:-1].split("\t")[4])

                bleu_score = nltk.translate.bleu_score.sentence_bleu(s1, s2)
                # sentence_bleu(s1, s2, smoothing_function=nltk.translate.bleu_score.SmoothingFunction.method1)

                self.data.append([s1, s2, int(label), [bleu_score]])
                if mode == "train":
                    self.data.append([s2, s1, int(label), [bleu_score]])

                local_max_length = max(len(s1), len(s2))
                if local_max_length > self.max_length:
                    self.max_length = local_max_length

            self.data_size = len(self.data)

    def is_available(self):
        if self.index < self.data_size:
            return True
        else:
            return False

    def next(self):
        self.index += 1
        if (self.is_available()):
            return self.data[self.index]
        else:
            return [None] * len(self.data[0])

    def next_batch(self):
        for i in range(self.batch_size):
            s1, s2, label, feature = self.next()

            if s1 is None:
                break

            # [1, d0, s]
            s1_mat = tf.expand_dims(tf.pad(tf.stack([self.word2vec.get(w) for w in s1], axis=1),
                                           [[0, 0], [0, self.max_length - len(s1)]]), 0)
            s2_mat = tf.expand_dims(tf.pad(tf.stack([self.word2vec.get(w) for w in s2], axis=1),
                                           [[0, 0], [0, self.max_length - len(s2)]]), 0)

            if i == 0:
                batch1 = s1_mat
                batch2 = s2_mat
                labels = [label]
                features = [feature]
            else:
                batch1 = tf.concat([s1_mat, batch1], axis=0)
                batch2 = tf.concat([s2_mat, batch2], axis=0)
                labels.append(label)
                features.append(feature)

        labels = tf.contrib.layers.one_hot_encoding(labels=labels, num_classes=self.num_classes)
        features = tf.constant(features)
        return batch1, batch2, labels, features