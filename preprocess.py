
import numpy as np
import nltk

def load_word_embeddings():
    word_embeddings = {}

    with open("wiki.ko.vec", "r", encoding="utf-8") as f:
        for line in f:
            word_embeddings[line.split()[0]] = np.array(line.split()[1:])

    return word_embeddings

word_embeddings = load_word_embeddings()

with open("msr_paraphrase_train.txt", "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        sentence1 = line[:-1].split("\t")[3]
        sentence2 = line[:-1].split("\t")[4]

        for word in nltk.word_tokenize(sentence1):
            if word not in word_embeddings:
                print("S1 - OOV:", word)

        for word in nltk.word_tokenize(sentence2):
            if word not in word_embeddings:
                print("S2 - OOV:", word)

"""
import tensorflow as tf
with tf.device("/cpu:0"):
    hello = tf.constant("Hello, Tensorflow!")
    sess = tf.Session()
    print(sess.run(hello))
"""