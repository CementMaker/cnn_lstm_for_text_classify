# -*- coding:utf-8 -*-
# !/usr/bin/env python

import pickle
import os
import jieba
import jieba.analyse
import random

from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn

label_list = ['Military', 'Economy', 'Culture', 'Sports', 'Auto', 'Medicine']

stopword = set()
fd = open('../stopwords.txt', 'r', encoding='gbk')
for line in fd:
    stopword.add(line.strip())


def remove_stop_word(article):
    new_article = []
    for word in article:
        if word not in stopword:
            new_article.append(word)
    return new_article


def soft_max_label(label):
    new_label = 6 * [0]
    index = label_list.index(label)
    new_label[index] = 1
    return new_label


def get_one_hot():
    all_context, all_labels = zip(*loading_corpus())
    vocab_processor = learn.preprocessing.VocabularyProcessor(1500, min_frequency=5)
    all_context = vocab_processor.fit_transform(all_context)
    print("number of words :", len(vocab_processor.vocabulary_))
    train_x, test_x, train_y, test_y = train_test_split(list(all_context), all_labels, test_size=0.01)

    print(train_x[0], train_y[0])

    pickle.dump((train_x, train_y), open("./pkl/train.pkl", "wb"))
    pickle.dump((test_x, test_y), open("./pkl/test.pkl", "wb"))


def delete_and_split(all_context, all_labels):
    new_data = []
    data = zip(all_context, all_labels)
    for context, label in data:
        string = ' '.join(list(jieba.cut(context)))
        new_data.append((string, soft_max_label(label)))
    return new_data
            

# load all document in corpus
def loading_corpus(path='../corpus'):
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


def get_batch(epoches, batch_size):
    train_x, train_y = pickle.load(open("./pkl/train.pkl", "rb"))
    data = list(zip(train_x, train_y))

    for epoch in range(epoches):
        random.shuffle(data)
        for batch in range(0, len(train_y), batch_size):
            if batch + batch_size < len(train_y):
                yield data[batch: (batch + batch_size)]


if __name__ == "__main__":
    number = 1
    get_one_hot()
    for index in get_batch(3, 300):
        number += 1
    print(number)
