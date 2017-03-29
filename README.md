
# ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs

#### !!!Caution!!!: Implementation is not complete. I hope having it done very soon!

This is the implementation of **ABCNN**, which is proposed by [Wenpeng Yin et al.](https://arxiv.org/pdf/1512.05193.pdf), on **Tensorflow**.  
It includes all 4 models below:
- BCNN
- ABCNN-1
- ABCNN-2
- ABCNN-3

### Note:
- Implementation is now only focusing on PI task with [MSRP(Microsoft Research Paraphrase)](https://www.microsoft.com/en-us/download/details.aspx?id=52398) corpus.  
- Because the original corpus doesn't have a validation set, **msr_paraphrase_dev.txt** was made by extracting 400 random cases from the training data as the article suggests.

## Specification
- **preprocess.py**: preprocess MSRP data and import word2vec to use.
- **train.py**: train a model with configs.
- **test.py**: test the trained model.
- **ABCNN.py**: Implementation of ABCNN models.
- MSRP_Corpus
    - **msr_paraphrase_train.txt**: MSRP training data.
    - **msr_paraphrase_dev.txt**: MSRP validation(dev) data.
    - **msr_paraphrase_test.txt**: MSRP test data.
- **models**: saved models available on Tensorflow.

## Development Environment
- OS: Windows 10 (64 bit)
- Language: Python 3.5.3
- CPU: Intel Xeon CPU E3-1231 v3 3.4 GHz
- RAM: 16GB
- GPU support: GTX 970
- Libraries:
    - **tensorflow** 1.0.1
    - numpy 1.12.0
    - gensim 1.0.1
    - NLTK 3.2.2
    - scikit-learn 0.18.1

## Requirements

This model is based on pre-trained Word2vec([GoogleNews-vectors-negative300.bin](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download)) by T.Mikolov et al.  
You should download this file and place it in the root folder.


## Execution

    Paramters
    --lr: learning rate
    --ws: window_size
    --l2_reg: l2_reg modifier
    --batch_size: batch_size
    --model_type: model type

> (training): python train.py --lr=0.01 --ws=3 --l2_reg=0.0003 --batch_size=64 --model_type="BCNN"  
> (test): python test.py --lr=0.01 --ws=3 --l2_reg=0.0003 --batch_size=64 --model_type="BCNN"

## MISC.
- [Original code by the author?](https://github.com/yinwenpeng/Answer_Selection/tree/master/src)
