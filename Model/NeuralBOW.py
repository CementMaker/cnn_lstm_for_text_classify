import tensorflow as tf


class NeuralBOW(object):
    def __init__(self, sentence_length, embedding_size, num_label, vocab_size):
        self.sentence = tf.placeholder(dtype=tf.int32, shape=[None, sentence_length], name='sentence')
        self.label = tf.placeholder(dtype=tf.int32, shape=[None, num_label], name='label')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32)

        with tf.name_scope("embedding"):
            w = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size], dtype=tf.float32))
            self.input_embedding = tf.nn.embedding_lookup(params=w, ids=self.sentence)

        with tf.name_scope("pooling"):
            self.pooling = tf.reduce_mean(self.input_embedding, axis=1)

        with tf.name_scope("full_connected_layer"):
            w = tf.Variable(tf.truncated_normal(shape=[embedding_size, num_label], dtype=tf.float32))
            b = tf.Variable(tf.truncated_normal(shape=[num_label]), dtype=tf.float32)
            self.logits = tf.nn.xw_plus_b(self.pooling, w, b)

        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(self.losses)

        with tf.name_scope("prediction"):
            self.predict = tf.argmax(self.logits, axis=1)
            self.actual_label = tf.argmax(self.label, axis=1)
            self.equal = tf.cast(x=tf.equal(self.predict, self.actual_label), dtype=tf.float32)
            self.accuracy = tf.reduce_mean(self.equal)


# nbow = NeuralBOW(sentence_length=60,
#                  embedding_size=150,
#                  vocab_size=20000,
#                  num_label=5)
