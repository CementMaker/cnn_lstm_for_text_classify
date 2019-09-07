import tensorflow as tf

from tensorflow.contrib import rnn


class lstm(object):
    def __init__(self, num_layers, sequence_length, embedding_size, vocab_size, rnn_size, num_classes):
        # 输入数据以及数据标签
        self.label = tf.placeholder(tf.int32, [None, num_classes], name="label")
        self.sentence = tf.placeholder(tf.int32, [None, sequence_length], name="input_b")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope('embeddingLayer'):
            W = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size]))
            embedded = tf.nn.embedding_lookup(W, self.sentence)
            inputs = tf.unstack(embedded, axis=1)

        with tf.name_scope('lstm_layer'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            # 使用 MultiRNNCell 类实现深层循环网络中每一个时刻的前向传播过程，num_layers 表示有多少层
            # cell = rnn.BasicLSTMCell(rnn_size)
            self.cell = rnn.MultiRNNCell([cell] * num_layers)
            self.outputs, _ = rnn.static_rnn(self.cell, inputs, dtype=tf.float32)
            self.feature = tf.concat(self.outputs, axis=1)

        with tf.name_scope("dropout"):
            self.feature = tf.nn.dropout(self.feature, self.dropout_keep_prob)

        with tf.name_scope('softmaxLayer'):
            w = tf.Variable(tf.truncated_normal(shape=[rnn_size * sequence_length, num_classes]))
            b = tf.Variable(tf.truncated_normal(shape=[num_classes]))
            self.logits = tf.nn.xw_plus_b(self.feature, w, b)

        # 损失函数，采用softmax交叉熵函数
        with tf.name_scope('loss'):
            self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.label)
            self.loss = tf.reduce_mean(self.losses)

        with tf.name_scope('predict'):
            self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


# model = lstm(num_layers=1,
#              sequence_length=60,
#              embedding_size=50,
#              vocab_size=120000,
#              rnn_size=50,
#              num_classes=2)
