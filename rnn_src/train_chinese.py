import tensorflow as tf
import datetime
import numpy as np

from lstm import *
from PreProcess import *

import matplotlib.pyplot as plt

global_loss = []
global_accuracy = []


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

x_train, y_train, x_dev, y_dev = get_one_hot('../文档')
batches = batch_iter(list(zip(x_train, y_train)), 200, 20, shuffle=True)


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        lstm = Model(num_layers=2,
                     seq_length=2000,
                     vocab_size=37000,
                     rnn_size=128,
                     label_size=7)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss=lstm.cost, global_step=global_step)
        sess.run(tf.global_variables_initializer())

        def train_step(batch, label):
            feed_dict = {
                lstm.input_data: batch,
                lstm.targets: label,
                lstm.dropout_keep_prob: 0.5
            }
            _, step, loss, accuracy = sess.run(
                [optimizer, global_step, lstm.cost, lstm.accuracy],
                feed_dict=feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {}, accuracy {}".format(time_str, step, loss, accuracy))

            global_loss.append(loss)
            global_accuracy.append(accuracy)

        def dev_step(batch, label):
            feed_dict = {
                lstm.input_data: batch,
                lstm.targets: label,
                lstm.dropout_keep_prob: 0.5
            }
            step, loss, accuracy = sess.run([global_step, lstm.cost, lstm.accuracy], feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, accuracy {}".format(time_str, step, loss, accuracy))


        for data in batches:
            x_train, y_train = zip(*data)
            train_step(x_train, y_train)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 100 == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev)
                print("")

        x = list(range(len(global_loss)))
        plt.plot(x, global_loss, 'r', label="loss")
        plt.xlabel("batches")
        plt.ylabel("loss")
        plt.savefig("loss_logits.png")
        plt.close()

        plt.plot(x, global_accuracy, 'b', label="accuracy")
        plt.xlabel("batches")
        plt.ylabel("accuracy")
        plt.savefig("accuracy_logits.png")
        plt.close()

