#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn


class Model(object):
    def __init__(self, num_layers, seq_length, embedding_size, vocab_size, rnn_size, label_size):
        # 输入数据以及数据标签
        self.input_data = tf.placeholder(tf.int64, [None, seq_length])
        self.labels = tf.placeholder(tf.int64, [None, label_size])
        self.dropout_keep_prob = tf.placeholder(tf.float32, )

        cell = rnn.BasicLSTMCell(rnn_size)
        # 使用 MultiRNNCell 类实现深层循环网络中每一个时刻的前向传播过程，num_layers 表示有多少层
        self.cell = rnn.MultiRNNCell([cell] * num_layers)
        self.l2_loss = tf.constant(0.0)

        with tf.name_scope('embeddingLayer'):
            # W : 词表（embedding 向量），后面用来训练.
            W = tf.get_variable('W', [vocab_size, embedding_size])
            embedded = tf.nn.embedding_lookup(W, self.input_data)

            # shape: (batch_size, seq_length, cell.input_size) => seq_length * (batch_size, cell.input_size)
            inputs = tf.split(embedded, seq_length, 1)

            # 根据第二维展开,维度从0开始
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
            # 删除所有大小为1的维度,删除[1]为要删除维度的参数

        # outputs是最后一层每个节点的输出
        # last_state是每层最后一个节点的输出。
        with tf.name_scope('lstm_layer'):
            self.outputs, self.final_state = rnn.static_rnn(self.cell, inputs, dtype=tf.float32)

        with tf.name_scope("pooling"):
            for index in range(len(self.outputs)):
                self.outputs[index] = tf.expand_dims(self.outputs[index], -1)
                self.outputs[index] = tf.expand_dims(self.outputs[index], -1)
            print(self.outputs[1])


        with tf.name_scope("pooling"):
            for index in range(len(self.outputs)):
                self.outputs[index] = tf.nn.max_pool(
                    self.outputs[index],
                    ksize=[1, rnn_size, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
            self.all_feature = tf.squeeze(tf.concat(self.outputs, 1), [2, 3])
            print(self.all_feature)

        # with tf.name_scope("dropout"):
        #     self.h_drop = tf.nn.dropout(self.outputs[-1], self.dropout_keep_prob)

        with tf.name_scope('softmaxLayer'):
            W = tf.get_variable('w', [seq_length, label_size])
            b = tf.get_variable('b', [label_size])
            logits = tf.nn.xw_plus_b(self.all_feature, W, b)
            self.probs = tf.nn.softmax(logits, dim=1)

        # 损失函数，采用softmax交叉熵函数
        with tf.name_scope('loss'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                               labels=self.labels))

        with tf.name_scope('predict'):
            self.correct_pred = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            # self.correct_num = tf.reduce_sum(tf.cast(self.correct_pred, tf.float32))

    def predict_label(self, sess, labels, text):
        x = np.array(text)
        feed = {self.input_data: x}
        probs, state = sess.run([self.probs, self.final_state], feed_dict=feed)

        results = np.argmax(probs, 1)
        id2labels = dict(zip(labels.values(), labels.keys()))
        labels = map(id2labels.get, results)
        return labels

    def predict_class(self, sess, text):
        x = np.array(text)
        feed = {self.input_data: x}
        probs, state = sess.run([self.probs, self.final_state], feed_dict=feed)
        results = np.argmax(probs, 1)
        return results


# lstm = Model(num_layers=1,
#              seq_length=500,
#              embedding_size=150,
#              vocab_size=74680,
#              rnn_size=150,
#              label_size=6)