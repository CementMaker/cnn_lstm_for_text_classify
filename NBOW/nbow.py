import tensorflow as tf


class nbow(object):
    def __init__(self, sentence_len, vocab_size, embedding_size, num_label):
        self.input = tf.placeholder(dtype=tf.int32, shape=[None, sentence_len])
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, num_label])
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32)

        with tf.name_scope("embedding"):
            w = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size]), dtype=tf.float32)
            self.input_embedding = tf.nn.embedding_lookup(params=w, ids=self.input)
            print(self.input_embedding)

        with tf.name_scope("mean_pooling"):
            self.feature = tf.reduce_mean(self.input_embedding, axis=1)

        with tf.name_scope("dropout"):
            self.feature_dropout = tf.nn.dropout(self.feature, keep_prob=self.dropout_keep_prob)

        with tf.name_scope("full_connected"):
            w = tf.Variable(tf.truncated_normal(shape=[embedding_size, num_label]), dtype=tf.float32)
            b = tf.Variable(tf.truncated_normal(shape=[num_label]), dtype=tf.float32)
            self.logits = tf.nn.xw_plus_b(self.feature, w, b)
            self.result = tf.nn.softmax(logits=self.logits, dim=1)

        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.result, labels=self.label)
            self.loss = tf.reduce_mean(self.losses)

        with tf.name_scope("prediction"):
            self.equal = tf.cast(tf.equal(tf.argmax(self.result, axis=1), tf.argmax(self.label, axis=1)), dtype=tf.float32)
            self.accuracy = tf.reduce_mean(self.equal)


# Nbow = nbow(sentence_len=1000,
#             vocab_size=20000,
#             embedding_size=128,
#             num_label=6)