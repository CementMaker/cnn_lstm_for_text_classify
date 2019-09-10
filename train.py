import os
import pickle
import random
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from Model import *
from preprocess import *
from sklearn.metrics import classification_report


# mini batch每个batch迭代
def get_batch(epoches, batch_size, train_x=None, train_y=None, seq_length=None):
    if train_x is None:
        train_x, train_y = pickle.load(open("./data/pkl/train.pkl", "rb"))
    data = list(zip(train_x, train_y, seq_length))

    for epoch in range(epoches):
        random.shuffle(data)
        for batch in range(0, len(train_y), batch_size):
            yield data[batch: (batch + batch_size)]


def train_step(model_train, batch, label):
    feed_dict = {
        model_train.model.sentence: batch,
        model_train.model.label: label,
        model_train.model.dropout_keep_prob: 0.5
    }
    _, summary, step, loss, accuracy, = model_train.sess.run(
        fetches=[model_train.optimizer, model_train.merged_summary_train,
                 model_train.global_step, model_train.model.loss,
                 model_train.model.accuracy],
        feed_dict=feed_dict)

    # 写入tensorBoard
    model_train.summary_writer_train.add_summary(summary, step)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {}, accuracy {}".format(time_str, step, loss, accuracy))


def dev_step(model_train, batch, label, return_predict=False):
    feed_dict = {
        model_train.model.sentence: batch,
        model_train.model.label: label,
        model_train.model.dropout_keep_prob: 1.0
    }
    summary, step, loss, accuracy, predict = model_train.sess.run(
        fetches=[model_train.merged_summary_test, model_train.global_step,
                 model_train.model.loss, model_train.model.accuracy,
                 model_train.model.predict],
        feed_dict=feed_dict)
    
    # 写入tensorBoard
    model_train.summary_writer_test.add_summary(summary, step)
    print("\t test: step {}, loss {}, accuracy {}".format(step, loss, accuracy))


class NeuralBowTrain(object):
    def __init__(self):
        self.sess = tf.Session()
        self.model = NeuralBOW.NeuralBOW(sentence_length=50,
                                         embedding_size=150,
                                         vocab_size=20005,
                                         num_label=6)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        
        # 定义optimizer
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.model.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
        self.batches = get_batch(5, 100)

        # tensorBoard
        tf.summary.scalar("loss", self.model.loss)
        tf.summary.scalar("accuracy", self.model.accuracy)
        self.merged_summary_train = tf.summary.merge_all()
        self.merged_summary_test = tf.summary.merge_all()
        self.summary_writer_train = tf.summary.FileWriter("./summary/nbow_summary/train", graph=self.sess.graph)
        self.summary_writer_test = tf.summary.FileWriter("./summary/nbow_summary/test", graph=self.sess.graph)


class textCnnTrain(object):
    def __init__(self):
        self.sess = tf.Session()
        self.model = textCnn.Cnn(sequence_length=50,
                                 embedding_size=100,
                                 filter_sizes=[1, 2, 3, 4],
                                 num_filters=10,
                                 num_classes=6,
                                 vocab_size=20005)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
         
        # 定义optimizer
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.model.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
        self.batches = get_batch(1, 400)

        # tensorBoard
        tf.summary.scalar('loss', self.model.loss)
        tf.summary.scalar('accuracy', self.model.accuracy)
        self.merged_summary_train = tf.summary.merge_all()
        self.merged_summary_test = tf.summary.merge_all()
        self.summary_writer_train = tf.summary.FileWriter("./summary/cnn_summary/train", graph=self.sess.graph)
        self.summary_writer_test = tf.summary.FileWriter("./summary/cnn_summary/test", graph=self.sess.graph)


class textRnnTrain(object):
    def __init__(self):
        self.sess = tf.Session()
        self.model = textLstm.lstm(num_layers=1,
                                   sequence_length=50,
                                   embedding_size=100,
                                   vocab_size=20005,
                                   rnn_size=100,
                                   num_classes=6)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # 定义optimizer
        self.optimizer = tf.train.AdamOptimizer(0.005).minimize(self.model.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
        self.batches = get_batch(1, 400)

        # tensorBoard
        tf.summary.scalar('loss', self.model.loss)
        tf.summary.scalar('accuracy', self.model.accuracy)
        self.merged_summary_train = tf.summary.merge_all()
        self.merged_summary_test = tf.summary.merge_all()
        self.summary_writer_train = tf.summary.FileWriter("./summary/rnn_summary/train", graph=self.sess.graph)
        self.summary_writer_test = tf.summary.FileWriter("./summary/rnn_summary/test", graph=self.sess.graph)


class textDynamicRnnTrain(object):
    def __init__(self):
        self.feature = DynamicRnnfeature()
        
        self.sess = tf.Session()
        self.model = TextDynamicRNN.lstm(layer_sizes=[128, 256],
                                         sequence_length=,
                                         embedding_size=100,
                                         vocab_size=20005,
                                         rnn_size=100,
                                         num_classes=6,
                                         max_length=3236)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # 定义optimizer
        self.optimizer = tf.train.AdamOptimizer(0.005).minimize(self.model.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
        self.batches = get_batch(1, 400, self.feature.train_x, self.feature.train_y)

        # tensorBoard
        tf.summary.scalar('loss', self.model.loss)
        tf.summary.scalar('accuracy', self.model.accuracy)
        self.merged_summary_train = tf.summary.merge_all()
        self.merged_summary_test = tf.summary.merge_all()
        self.summary_writer_train = tf.summary.FileWriter("./summary/dynamic_rnn_summary/train", graph=self.sess.graph)
        self.summary_writer_test = tf.summary.FileWriter("./summary/dynamic_rnn_summary/test", graph=self.sess.graph)


def main(model_train):
    try:
        test_x, test_y = model_train.feature.test_x, model_train.feature.test_y
    except:
        test_x, test_y = pickle.load(open("./data/pkl/test.pkl", "rb"))
    for data in model_train.batches:
        x_train, y_train = zip(*data)
        train_step(model_train, x_train, y_train)
        current_step = tf.train.global_step(model_train.sess, model_train.global_step)
        if current_step % 10 == 0:
            print("dev step\t:")
            dev_step(model_train, test_x, test_y)

    dev_step(model_train, test_x, test_y, return_predict=True)
    # print(classification_report(y_true=test_y, y_pred=predict))


if __name__ == "__main__":
    # Net = NeuralBowTrain()
    # Net = textCnnTrain()
    # Net = textRnnTrain()
    Net = textDynamicRnnTrain()
    main(Net)

