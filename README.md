
# ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs

This is the implementation of **ABCNN**, which is proposed by [Wenpeng Yin et al.](https://arxiv.org/pdf/1512.05193.pdf), on **Tensorflow**.
It includes all 4 models below:
- BCNN
- ABCNN-1
- ABCNN-2
- ABCNN-3

Note that implementation is now only focusing on PE task with [MSRP(Microsoft Research Paraphrase)](https://www.microsoft.com/en-us/download/details.aspx?id=52398) corpus.

## Specification
- **preprocess.py**: preprocess MSRP data and import word2vec to use.
- **train.py**: run the model with configs.
- **ABCNN.py**: Implementation of ABCNN models.
- **msr_paraphrase_train.txt**: MSRP(task: PE) training data.
- **msr_paraphrase_dev.txt**: MSRP(task: PE) validation(dev) data.
- **msr_paraphrase_test.txt**: MSRP(task: PE) test data.

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
    - numpy

## Requirements

This model is based on pre-trained Word2vec([GoogleNews-vectors-negative300.bin](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download)) by T.Mikolov et al.
You should download this file and place it in the root folder.


## Execution

