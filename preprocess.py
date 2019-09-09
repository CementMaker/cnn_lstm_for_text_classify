import os
import sys
import pickle
import logging
import jieba

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')
logger = logging.getLogger(__name__)

class NNfeature(object):
    def __init__(self, data_path='./data/context'):
        logger.info('NNfeature loading corpus ...')

        # 枚举所有的文件
        jieba.enable_parallel(4)
        self.context, self.label = [], []
        for file in tqdm(os.listdir(path=data_path)):
            try:
                label = file.split('_')[0]
                filePath = os.path.join(data_path, file)
                with open(filePath, 'r', encoding='utf-8') as fd:
                    context = fd.read()
                self.context.append(context)
                self.label.append(label)
            except:
                logger.warning('file %s have some problem ...' % file)

        self.label_list = ['Military', 'Economy', 'Culture', 'Sports', 'Auto', 'Medicine']
        self.label = [self.label_list.index(label) for label in self.label]

        self.one_hot_label = np.zeros(shape=(len(self.label), 6))
        self.one_hot_label[np.arange(0, len(self.label)), self.label] = 1
        logger.debug('one hot label shape: (%d, %d)' % (self.one_hot_label.shape))

        self.context = [' '.join(list(jieba.cut(context))) for context in tqdm(self.context)]
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=20000, # 词表去20000，词表的提取根据TF的计算结果排序
            filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ',
            lower=True,
            split=' ',
            char_level=False,
            oov_token=None,
            document_count=0)

        self.tokenizer.fit_on_texts(self.context)
        self.context = np.array(self.tokenizer.texts_to_sequences(self.context))
        self.context = pad_sequences(self.context, maxlen=50, padding='post')
        logger.debug('context idx shape: (%d, %d)' % (self.context.shape))

        self.train_x, self.test_x, self.train_y, self.test_y = \
            train_test_split(self.context, self.one_hot_label, test_size=0.05)
        logger.debug('self.train_x shape: (%d, %d)' % (self.train_x.shape))
        logger.debug('self.test_x shape: (%d, %d)' % (self.test_x.shape))
        logger.debug('self.train_y shape: (%d, %d)' % (self.train_y.shape))
        logger.debug('self.test_y shape: (%d, %d)' % (self.test_y.shape))

        pickle.dump((self.test_x, self.test_y), open("./data/pkl/test.pkl", "wb"))
        pickle.dump((self.train_x, self.train_y), open("./data/pkl/train.pkl", "wb"))


class fastTextfeature(object):
    def __init__(self, data_path='./data/context'):
        logger.info('fastTextfeature loading corpus ...')

        # 枚举所有的文件
        jieba.enable_parallel(4)
        self.context, self.label = [], []
        for file in tqdm(os.listdir(path=data_path)):
            try:
                label = file.split('_')[0]
                filePath = os.path.join(data_path, file)
                with open(filePath, 'r', encoding='utf-8') as fd:
                    context = fd.read()
                self.context.append(context)
                self.label.append('__' + label + '__')
            except:
                logger.warning('file %s have some problem ...' % file)
        self.context = [' '.join(list(jieba.cut(context))) for context in tqdm(self.context)]
        self.train_context, self.test_context, self.train_label, self.test_label =\
            train_test_split(self.context, self.label, test_size=0.05)

        train_data_fd = open('./data/fastTextData/train_data', 'w+')
        for label, context in zip(self.train_label, self.train_context):
            train_data_fd.write(label + '\t' + context)
        train_data_fd.close()

        valid_data_fd = open('./data/fastTextData/valid_data', 'w+')
        for label, context in zip(self.test_label, self.test_context):
            valid_data_fd.write(label + '\t' + context)
        valid_data_fd.close()

        logger.debug('self.train_context shape: %d' % len(self.train_context))
        logger.debug('self.test_context shape: %d' % len(self.test_context))
        logger.debug('self.train_label shape: %d' % len(self.train_label))
        logger.debug('self.test_label shape: %d' % len(self.test_label))


if __name__ == '__main__':
    f = NNfeature()
    # f = fastTextfeature()

