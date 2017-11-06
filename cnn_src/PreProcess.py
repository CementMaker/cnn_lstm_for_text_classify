# -*- coding:utf-8 -*-
# !/usr/bin/env python

import pickle
import os
import time
import datetime
import jieba
import jieba.analyse
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
from collections import Counter

# label set
label_list = ['Military', 'Economy', 'Culture', 'Sports', 'Auto', 'Medicine']


# get label for softmax
def soft_max_label(label):
    new_label = 6 * [0]
    index = label_list.index(label)
    new_label[index] = 1
    return new_label


# encode every word, every will encode with index
def get_one_hot(path):
    all_context, all_labels = zip(*loading_corpus(path))
    vocab_processor = learn.preprocessing.VocabularyProcessor(1500, min_frequency=5)
    all_context = list(vocab_processor.fit_transform(all_context))
    print("number of words :", len(vocab_processor.vocabulary_))
    
    train_x, test_x, train_y, test_y = train_test_split(all_context, all_labels, test_size=0.01)
    pickle.dump((test_x, test_y), open("corpus_test.pkl", "wb"))
    pickle.dump((train_x, train_y), open("corpus_train.pkl", "wb"))

# cut sentence and join
def delete_and_split(all_context, all_labels):
    new_data = []
    data = zip(all_context, all_labels)
    for context, label in data:
        article = ' '.join(list(jieba.cut(context)))
        new_data.append((article, soft_max_label(label)))
    return new_data


# load all document in corpus
def loading_corpus(path):
    allContext = []
    allLabel = []
    allFile = os.listdir(path=path)
    for file in allFile:
        label = file.split('_')[0]
        filePath = os.path.join(path, file)
        with open(filePath, 'r', encoding='utf-8') as fd:
            context = fd.read()
        allContext.append(context)
        allLabel.append(label)

    newData = delete_and_split(allContext, allLabel)
    return newData


# get batch for CNN
def get_batch(epoches, batch_size):
    train_x, train_y = pickle.load(open("corpus_train.pkl", "rb"))
    data = list(zip(train_x, train_y))
    random.shuffle(data)
    for epoch in range(epoches):
        for batch in range(0, len(data), batch_size):
            if batch + batch_size < len(data):
                yield data[batch: (batch + batch_size)]


if __name__ == "__main__":
    get_one_hot("../corpus")