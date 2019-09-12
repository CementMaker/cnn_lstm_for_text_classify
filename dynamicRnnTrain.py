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


def get_batch(epoches, batch_size, train_x=None, train_y=None, seq_length=None):
    data = list(zip(train_x, train_y, seq_length))
    for epoch in range(epoches):
        random.shuffle(data)
        for batch in range(0, len(train_y), batch_size):
            yield data[batch: (batch + batch_size)]


def train_step(model_train, batch, sequence_length, label):
    feed_dict = {
        model_train.model.sentence: batch,
        model_train.model.label: label,
        model_train.model.sentence_length: sequence_length,
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


def dev_step(model_train, batch, sequence_length, label):
    feed_dict = {
        model_train.model.sentence: batch,
        model_train.model.label: label,
        model_train.model.sentence_length: sequence_length,
        model_train.model.dropout_keep_prob: 1.0
    }
    summary, step, loss, accuracy = model_train.sess.run(
        fetches=[model_train.merged_summary_test, model_train.global_step,
                 model_train.model.loss, model_train.model.accuracy],
        feed_dict=feed_dict)
    
    # 写入tensorBoard
    model_train.summary_writer_test.add_summary(summary, step)
    print("\t test: step {}, loss {}, accuracy {}".format(step, loss, accuracy))


class textDynamicRnnTrain(object):
    def __init__(self):
        self.feature = DynamicRnnfeature()
        
        self.sess = tf.Session()
        self.model = TextDynamicRNN.lstm(layer_sizes=[128, 256],
                                         embedding_size=100,
                                         vocab_size=20005,
                                         rnn_size=100,
                                         num_classes=6,
                                         max_length=3236)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # 定义optimizer
        self.optimizer = tf.train.AdamOptimizer(0.005).minimize(self.model.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
        self.batches = get_batch(1, 400, self.feature.train_x, self.feature.train_y, self.feature.seq_len_train)

        # tensorBoard
        tf.summary.scalar('loss', self.model.loss)
        tf.summary.scalar('accuracy', self.model.accuracy)
        self.merged_summary_train = tf.summary.merge_all()
        self.merged_summary_test = tf.summary.merge_all()
        self.summary_writer_train = tf.summary.FileWriter("./summary/dynamic_rnn_summary/train", graph=self.sess.graph)
        self.summary_writer_test = tf.summary.FileWriter("./summary/dynamic_rnn_summary/test", graph=self.sess.graph)


def main(model_train):
    for data in model_train.batches:
        x_train, y_train, seq_len = zip(*data)
        train_step(model_train, x_train, y_train, seq_len)
        current_step = tf.train.global_step(model_train.sess, model_train.global_step)
        if current_step % 10 == 0:
            print("dev step\t:", end='')
            dev_step(model_train, model_train.feature.test_x, model.feature.seq_len_test, model_train.feature.test_y)

    dev_step(model_train, model_train.feature.test_x, model.feature.seq_len_test, model_train.feature.test_y)


if __name__ == "__main__":
    Net = textDynamicRnnTrain()
    main(Net)

