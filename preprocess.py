
import numpy as np
import nltk
import time
import random

import gensim


"""
def load_word_embeddings():
    word_embeddings = {}

    with open("wiki.ko.vec", "r", encoding="utf-8") as f:
        for line in f:
            word_embeddings[line.split()[0]] = np.array(line.split()[1:])

    return word_embeddings

start_time = time.time()
print("loading embeddings...")
word_embeddings = load_word_embeddings()
print("embeddings loaded!", time.time() - start_time)

unknown_embedding = np.random.uniform(-0.01,0.01,300)
"""

with open("msr_paraphrase_train.txt", "r", encoding="utf-8") as f:
    f.readline()
    c = 0
    cc = 0
    for line in f:
        sentence1 = line[:-1].split("\t")[3]
        sentence2 = line[:-1].split("\t")[4]
            
        for word in nltk.word_tokenize(sentence1):
            cc += 1
            if word.lower() not in word_embeddings:
                print("S1 - OOV:", word)
                c += 1

        for word in nltk.word_tokenize(sentence2):
            cc += 1
            if word.lower() not in word_embeddings:
                print("S2 - OOV:", word)
                c += 1

    print(c, cc, c/cc)

"""
import tensorflow as tf
with tf.device("/cpu:0"):
    hello = tf.constant("Hello, Tensorflow!")
    sess = tf.Session()
    print(sess.run(hello))
"""
