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
    def __init__(self, mode, word2vec, max_len=0, parsing_method="normal"):
        self.s1s, self.s2s, self.labels, self.features = [], [], [], []
        self.index, self.max_len, self.word2vec = 0, max_len, word2vec

        with open("./MSRP_Corpus/msr_paraphrase_" + mode + ".txt", "r", encoding="utf-8") as f:
            f.readline()

            for line in f:
                label = int(line[:-1].split("\t")[0])
                if parsing_method == "NLTK":
                    s1 = nltk.word_tokenize(line[:-1].split("\t")[3])
                    s2 = nltk.word_tokenize(line[:-1].split("\t")[4])
                else:
                    s1 = line[:-1].split("\t")[3].strip().split()
                    s2 = line[:-1].split("\t")[3].strip().split()

                bleu_score = nltk.translate.bleu_score.sentence_bleu(s1, s2)
                # sentence_bleu(s1, s2, smoothing_function=nltk.translate.bleu_score.SmoothingFunction.method1)

                self.s1s.append(s1)
                self.s2s.append(s2)
                self.labels.append(label)
                self.features.append([bleu_score, len(s1), len(s2)])

                if mode == "train":
                    self.s1s.append(s2)
                    self.s2s.append(s1)
                    self.labels.append(label)
                    self.features.append([bleu_score, len(s2), len(s1)])

                local_max_len = max(len(s1), len(s2))
                if local_max_len > self.max_len:
                    self.max_len = local_max_len

        self.data_size = len(self.s1s)
        self.num_features= len(self.features[0])

        # shuffle data
        """
        r_indexes = list(range(self.data_size))
        self.s1s = [self.s1s[i] for i in r_indexes]
        self.s2s = [self.s2s[i] for i in r_indexes]
        self.labels = [self.labels[i] for i in r_indexes]
        self.features = [self.features[i] for i in r_indexes]
        """

    def is_available(self):
        if self.index < self.data_size:
            return True
        else:
            return False

    def next(self):
        if (self.is_available()):
            self.index += 1
            return self.data[self.index - 1]
        else:
            return

    def next_batch(self, batch_size, num_classes):
        batch_size = min(self.data_size - self.index, batch_size)
        s1_mats, s2_mats = [], []

        for i in range(batch_size):
            s1 = self.s1s[self.index + i]
            s2 = self.s2s[self.index + i]

            # [1, d0, s]
            s1_mats.append(tf.expand_dims(tf.pad(tf.stack([self.word2vec.get(w) for w in s1], axis=1),
                                                 [[0, 0], [0, self.max_len - len(s1)]]), 0))
            s2_mats.append(tf.expand_dims(tf.pad(tf.stack([self.word2vec.get(w) for w in s2], axis=1),
                                                 [[0, 0], [0, self.max_len - len(s2)]]), 0))

        batch_s1s = tf.concat(s1_mats, axis=0)
        batch_s2s = tf.concat(s2_mats, axis=0)
        batch_labels = tf.contrib.layers.one_hot_encoding(labels=self.labels[self.index:self.index + batch_size],
                                                          num_classes=num_classes)
        batch_features = tf.constant(self.features[self.index:self.index + batch_size])

        self.index += batch_size

        return batch_s1s, batch_s2s, batch_labels, batch_features

    def reset_index(self):
        self.index = 0
