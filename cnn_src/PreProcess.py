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


# stop words
stopword = set()
fd = open('../stopwords.txt', 'r', encoding='gbk')
for line in fd:
    stopword.add(line.strip())


# 判断是否是汉子
def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'/u4e00' and uchar <= u'/u9fa5':
        return True
    else:
        return False


# 判断词汇是不是只有中文
def is_chinese_word(string):
    for word in string:
        if is_chinese(word) is False:
            return False
    return True


# remove stop words
def remove_stop_word(article):
    new_article = []
    for word in article:
        if word not in stopword and is_chinese_word(word):
            new_article.append(word)
    return new_article


# get label for softmax
def soft_max_label(label):
    new_label = 6 * [0]
    index = label_list.index(label)
    new_label[index] = 1
    return new_label


# encode every word
def get_one_hot(path):
    all_context, all_labels = zip(*loading_corpus())
    vocab_processor = learn.preprocessing.VocabularyProcessor(1500, min_frequency=5)
    all_context = list(vocab_processor.fit_transform(all_context))
    print("number of words :", len(vocab_processor.vocabulary_))
    return all_context, all_labels


# cut sentence and join
def delete_and_split(all_context, all_labels):
    new_data = []
    data = zip(all_context, all_labels)
    for context, label in data:
        article = ' '.join(list(jieba.cut(context)))
        new_data.append((article, soft_max_label(label)))
    print(new_data[1:20])
    return new_data
            

# load all document in 文档
def loading_data_set(path):
    all_context = []
    all_labels = []
    all_directory = os.listdir(path)
    for directory in all_directory:
        all_file = os.listdir(os.path.join(path, directory))
        print("路径 = ", directory, "时间：", time.asctime((time.localtime(time.time()))))
        for file in all_file:
            with open(os.path.join(path, directory, file), 'r', encoding='gbk') as fd:
                context = fd.read()
            all_context.append(context)
            all_labels.append(directory)

    print("分词开始时间：", time.asctime((time.localtime(time.time()))))
    new_data = delete_and_split(all_context, all_labels)
    print("分词结束时间：", time.asctime((time.localtime(time.time()))))
    pickle.dump(new_data, open("new_data.pkl", "wb"))
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


# get batch for DNN
def get_batch(epoches, batch_size):
    # all_context, all_labels = get_one_hot('../corpus')
    # pickle.dump((all_context, all_labels), open("corpus.pkl", "wb"))
    # all_context, all_labels = pickle.load(open("corpus.pkl", "rb"))
    # train_x, test_x, train_y, test_y = train_test_split(all_context, all_labels, test_size=0.01)

    # pickle.dump((test_x, test_y), open("corpus_test.pkl", "wb"))
    # pickle.dump((train_x, train_y), open("corpus_train.pkl", "wb"))

    train_x, train_y = pickle.load(open("corpus_train.pkl", "rb"))
    data = list(zip(train_x, train_y))
    random.shuffle(data)
    for epoch in range(epoches):
        for batch in range(0, len(data), batch_size):
            if batch + batch_size < len(data):
                yield data[batch: (batch + batch_size)]


number = 1
for index in get_batch(3, 300):
    print(number)
    number += 1



# if __name__ == "__main__":
#     get_one_hot("../corpus")
#     print("hello world")