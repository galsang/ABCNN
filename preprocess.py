
import numpy as np
import nltk
import gensim

class Word2Vec():
    def __init__(self):
        # Load Google's pre-trained Word2Vec model.
        self.model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
        self.unknowns = np.random.uniform(-0.01,0.01,300).astype("float32")

    def get(self, word):
        if word not in self.model.vocab:
            return self.unknowns
        else:
            return self.model.word_vec(word)

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