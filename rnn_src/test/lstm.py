#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn


class Model(object):
    def __init__(self, num_layers, seq_length, embedding_size, vocab_size, rnn_size, label_size):
        self.input_x = tf.placeholder(tf.int64, [None, seq_length])
        self.input_y = tf.placeholder(tf.int64, [None, label_size])
        self.dropout_keep_prob = tf.placeholder(tf.float32, )

        with tf.name_scope('embeddingLayer'):
            W = tf.get_variable('W', [vocab_size, embedding_size])
            embedded = tf.nn.embedding_lookup(W, self.input_x)

            # shape: (batch_size, seq_length, cell.input_size) => seq_length * (batch_size, cell.input_size)
            inputs = tf.split(embedded, seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # outputs是最后一层每个节点的输出
        # last_state是每层最后一个节点的输出。
        with tf.name_scope('lstm_layer'):
            cell = rnn.BasicLSTMCell(rnn_size)
            # 使用 MultiRNNCell 类实现深层循环网络中每一个时刻的前向传播过程,num_layers 表示有多少层
            self.cell = rnn.MultiRNNCell([cell] * num_layers)
            self.outputs, _ = rnn.static_rnn(self.cell, inputs, dtype=tf.float32)

        with tf.name_scope('softmaxLayer'):
            W = tf.get_variable('w', [rnn_size, label_size])
            b = tf.get_variable('b', [label_size])
            logits = tf.nn.xw_plus_b(self.outputs[-1], W, b)
            self.probs = tf.nn.softmax(logits, dim=1)

        # 损失函数，采用softmax交叉熵函数
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y))

        with tf.name_scope('predict'):
            self.correct_pred = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
