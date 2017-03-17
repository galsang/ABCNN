
import numpy as np
import nltk
import gensim

# Load Google's pre-trained Word2Vec model.
word2vec = gensim.models.Word2Vec.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
unknowns = np.random.uniform(-0.01,0.01,300)

with open("msr_paraphrase_train.txt", "r", encoding="utf-8") as f:
    f.readline()
    c = 0
    cc = 0
    for line in f:
        sentence1 = line[:-1].split("\t")[3]
        sentence2 = line[:-1].split("\t")[4]
            
        for word in nltk.word_tokenize(sentence1):
            cc += 1
            if word not in word2vec.vocab:
                unknowns
            else:
                word2vec[word]


                c += 1

        for word in nltk.word_tokenize(sentence2):
            cc += 1
            if (word  not in word2vec.vocab):
                #print("S2 - OOV:", word)
                c += 1

    print(c, cc, c/cc)

"""
import tensorflow as tf
with tf.device("/cpu:0"):
    hello = tf.constant("Hello, Tensorflow!")
    sess = tf.Session()
    print(sess.run(hello))
"""
