#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn


class lstm(object):
    def __init__(self, num_layers, seq_length, embedding_size, vocab_size, rnn_size, label_size):
        # 输入数据以及数据标签
        self.input_x = tf.placeholder(tf.int64, [None, seq_length])
        self.input_y = tf.placeholder(tf.int64, [None, label_size])
        self.dropout_keep_prob = tf.placeholder(tf.float32, )


        with tf.name_scope('embeddingLayer'):
            # W : 词表（embedding 向量），后面用来训练.
            W = tf.get_variable('W', [vocab_size, embedding_size])
            embedded = tf.nn.embedding_lookup(W, self.input_x)
            inputs = tf.unstack(embedded, axis=1)

        with tf.name_scope('lstm_layer'):
            cell = rnn.BasicLSTMCell(rnn_size)
            # 使用 MultiRNNCell 类实现深层循环网络中每一个时刻的前向传播过程，num_layers 表示有多少层
            self.cell = rnn.MultiRNNCell([cell] * num_layers)
            self.outputs, self.final_state = rnn.static_rnn(self.cell, inputs, dtype=tf.float32)

        with tf.name_scope('softmaxLayer'):
            W = tf.get_variable('w', [rnn_size, label_size])
            b = tf.get_variable('b', [label_size])
            logits = tf.nn.xw_plus_b(self.outputs[-1], W, b)
            self.probs = tf.nn.softmax(logits, dim=1)

        # 损失函数，采用softmax交叉熵函数
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                               labels=self.input_y))

        with tf.name_scope('predict'):
            self.correct_pred = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            # self.correct_num = tf.reduce_sum(tf.cast(self.correct_pred, tf.float32))

    def predict_label(self, sess, labels, text):
        x = np.array(text)
        feed = {self.input_x: x}
        probs, state = sess.run([self.probs, self.final_state], feed_dict=feed)

        results = np.argmax(probs, 1)
        id2labels = dict(zip(labels.values(), labels.keys()))
        labels = map(id2labels.get, results)
        return labels

    def predict_class(self, sess, text):
        x = np.array(text)
        feed = {self.input_x: x}
        probs, state = sess.run([self.probs, self.final_state], feed_dict=feed)
        results = np.argmax(probs, 1)
        return results


class Blstm(object):
    def __init__(self, num_layers, seq_length, embedding_size, vocab_size, rnn_size, label_size):
        # 输入数据以及数据标签
        self.input_x = tf.placeholder(tf.int32, [None, seq_length], name="input_x1")
        self.input_y = tf.placeholder(tf.float32, [None, label_size], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.l2_loss = tf.constant(0.0)

        with tf.name_scope('embeddingLayer'):
            w = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
            embedded = tf.nn.embedding_lookup(w, self.input_x)
            inputs = tf.unstack(embedded, axis=1)

        with tf.name_scope("fw"):
            stacked_rnn_fw = []
            for _ in range(num_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
                stacked_rnn_fw.append(fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw"):
            stacked_rnn_bw = []
            for _ in range(num_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
                stacked_rnn_bw.append(bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)

        with tf.name_scope("output"):
            outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, inputs, dtype=tf.float32)

        with tf.name_scope("result"):
            w = tf.Variable(tf.random_uniform([2 * rnn_size, label_size], -1.0, 1.0), name='W')
            b = tf.get_variable('b', [label_size])
            self.output = tf.nn.xw_plus_b(outputs[-1], w, b)
            self.logits = tf.nn.softmax(self.output, dim=1)

        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(self.losses)

        with tf.name_scope("accuracy"):
            self.accuracys = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.input_y, axis=1), name="equal")
            self.accuracy = tf.reduce_mean(tf.cast(self.accuracys, "float"), name="accuracy")


class dynamic_rnn(object):
    def __init__(self, num_layers, seq_length, embedding_size, vocab_size, rnn_size, label_size):
        # 输入数据以及数据标签
        self.input_x = tf.placeholder(tf.int64, [None, seq_length])
        self.input_y = tf.placeholder(tf.int64, [None, label_size])
        self.dropout_keep_prob = tf.placeholder(tf.float32, )

        with tf.name_scope('embeddingLayer'):
            W = tf.get_variable('W', [vocab_size, embedding_size])
            embedded = tf.nn.embedding_lookup(W, self.input_x)

        with tf.name_scope("output"):
            stacked_rnn = []
            for _ in range(num_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
                stacked_rnn.append(fw_cell)
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
            self.outputs, self.final_state = tf.nn.dynamic_rnn(lstm_cell, embedded, dtype=tf.float32, time_major=False)
            self.out = tf.unstack(tf.nn.batch_normalization(self.outputs), axis=1)[-1]

        with tf.name_scope('softmax_Layer'):
            W = tf.get_variable('w', [rnn_size, label_size])
            b = tf.get_variable('b', [label_size])
            logits = tf.nn.xw_plus_b(self.out, W, b)
            self.probs = tf.nn.softmax(logits, dim=1)

        # 损失函数，采用softmax交叉熵函数
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                               labels=self.input_y))

        with tf.name_scope('predict'):
            self.correct_pred = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            self.correct_num = tf.reduce_sum(tf.cast(self.correct_pred, tf.float32))

# model = dynamic_rnn(num_layers=1,
#                     seq_length=10,
#                     embedding_size=20,
#                     vocab_size=40,
#                     rnn_size=50,
#                     label_size=2)