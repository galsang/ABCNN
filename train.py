
import tensorflow as tf

from preprocess import Word2Vec
import BCNN

def main():
    word2vec = Word2Vec()

    s1 = "I am a boy"
    s2 = "you are a girl"

    s1_mat =  tf.concat([word2vec.get(w) for w in s1], 1)
    s2_mat = tf.concat([word2vec.get(w) for w in s2], 1)

    print(tf.shape(s1_mat))
    print(tf.shapes(s2_mat))

    return

    model = BCNN(s=4, w=2, lr=0, l2_reg=0)

    with tf.session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

main()