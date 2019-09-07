import tensorflow as tf


class Cnn(object):
    def __init__(self, sequence_length, vocab_size, embedding_size, filter_sizes, num_filters, num_classes):
        self.label = tf.placeholder(tf.int32, [None, num_classes], name="label")
        self.sentence = tf.placeholder(tf.int32, [None, sequence_length], name="input_b")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("embedding"):
            filter_shape = [vocab_size, embedding_size]
            w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(params=w, ids=self.sentence)
            self.embedded_chars_expand = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-max-pool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                conv = tf.nn.conv2d(input=self.embedded_chars_expand,
                                    filter=W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID")

                pooled = tf.nn.max_pool(
                    value=conv,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("full_connected_layer"):
            w = tf.Variable(tf.truncated_normal(shape=[num_filters_total, num_classes], stddev=0.1))
            b = tf.Variable(tf.truncated_normal(shape=[num_classes]))
            self.logits = tf.nn.xw_plus_b(self.h_pool_flat, w, b)
            self.predict = tf.argmax(self.logits, axis=1)

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predict, tf.argmax(self.label, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
