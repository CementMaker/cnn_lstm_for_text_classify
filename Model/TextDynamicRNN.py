import tensorflow as tf

from tensorflow.contrib import rnn


class lstm(object):
    def __init__(self, layer_sizes, embedding_size, vocab_size, rnn_size, num_classes, max_length):
        # 输入数据以及数据标签
        self.label = tf.placeholder(tf.int32, [None, num_classes], name="label")
        self.sentence = tf.placeholder(tf.int32, [None, max_length], name="input_b")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.sequence_length = tf.placeholder(tf.int32, [None,], name="sequence_length")

        with tf.name_scope('embeddingLayer'):
            W = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size]))
            self.embedding = tf.nn.embedding_lookup(W, self.sentence)

        with tf.name_scope('lstm_layer'):
            # 使用 MultiRNNCell 类实现深层循环网络中每一个时刻的前向传播过程
            self.rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in layer_sizes]
            self.multi_rnn_cell = rnn.MultiRNNCell(self.rnn_layers)
            self.outputs, self.rnn_state = tf.nn.dynamic_rnn(cell=self.multi_rnn_cell,
                                                             inputs=self.embedding,
                                                             time_major=False,
                                                             sequence_length=self.sequence_length,
                                                             dtype=tf.float32)
            print(len(self.rnn_state), len(self.rnn_state[0]), self.rnn_state[0][0])
            self.feature = tf.unstack(self.outputs, axis=1)[-1]

        with tf.name_scope('softmaxLayer'):
            # w = tf.Variable(tf.truncated_normal(shape=[rnn_size * sequence_length, num_classes]))
            # b = tf.Variable(tf.truncated_normal(shape=[num_classes]))
            w = tf.Variable(tf.truncated_normal(shape=[256, num_classes]))
            b = tf.Variable(tf.truncated_normal(shape=[num_classes]))
            self.logits = tf.nn.xw_plus_b(self.feature, w, b)

        # 损失函数，采用softmax交叉熵函数
        with tf.name_scope('loss'):
            self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.label)
            self.loss = tf.reduce_mean(self.losses)

        with tf.name_scope('predict'):
            self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            self.predict = tf.argmax(self.logits, axis=1)

