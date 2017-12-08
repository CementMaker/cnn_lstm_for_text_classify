import tensorflow as tf
from tensorflow.contrib import rnn

"""
 rnn.static_bidirectional_rnn :
 
 Creates a bidirectional recurrent neural network.

 Similar to the unidirectional case above (rnn) but takes input and builds
 independent forward and backward RNNs with the final forward and backward
 outputs depth-concatenated, such that the output will have the format
 [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
 forward and backward cell must match. The initial state for both directions
 is zero by default (but can be set optionally) and no intermediate states are
 ever returned -- the network is fully unrolled for the given (passed in)
 length(s) of the sequence(s) or completely unrolled if length(s) is not given.

 Args:
   cell_fw: An instance of RNNCell, to be used for forward direction.
   cell_bw: An instance of RNNCell, to be used for backward direction.
   inputs: A length T list of inputs, each a tensor of shape
     [batch_size, input_size], or a nested tuple of such elements.
   initial_state_fw: (optional) An initial state for the forward RNN.
     This must be a tensor of appropriate type and shape
     `[batch_size, cell_fw.state_size]`.
     If `cell_fw.state_size` is a tuple, this should be a tuple of
     tensors having shapes `[batch_size, s] for s in cell_fw.state_size`.
   initial_state_bw: (optional) Same as for `initial_state_fw`, but using
     the corresponding properties of `cell_bw`.
   dtype: (optional) The data type for the initial state.  Required if
     either of the initial states are not provided.
   sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
     containing the actual lengths for each of the sequences.
   scope: VariableScope for the created subgraph; defaults to
     "bidirectional_rnn"
"""


class Model(object):
    def BiRNN(self, x, scope, embedding_size, hidden_units, sequence_length):
        # Reshape to (n_steps * batch_size, n_input)
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        n_input, n_steps, n_hidden, n_layers = embedding_size, sequence_length, hidden_units, 1
        x = tf.split(tf.reshape(tf.transpose(x, [1, 0, 2]), [-1, n_input]), n_steps, 0)
        print(len(x))
        print(x[0])

        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                stacked_rnn_fw.append(fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                stacked_rnn_bw.append(bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
        print(outputs[-1])
        return outputs[-1]

    def __init__(self, num_layers, seq_length, embedding_size, vocab_size, rnn_size, label_size):
        # 输入数据以及数据标签
        self.input_x = tf.placeholder(tf.int32, [None, seq_length], name="input_x1")
        self.input_y = tf.placeholder(tf.float32, [None, label_size], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.l2_loss = tf.constant(0.0)

        with tf.name_scope('embeddingLayer'):
            # w : 词表（embedding 向量），后面用来训练.
            w = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
            embedded = tf.nn.embedding_lookup(w, self.input_x)

            # 根据第二维展开,维度从0开始
            # 删除所有大小为1的维度,删除[1]为要删除维度的参数
            inputs = tf.split(embedded, seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # outputs是最后一层每个节点的输出
        # last_state是每层最后一个节点的输出。
        with tf.name_scope("output"):
            w = tf.Variable(tf.random_uniform([2 * rnn_size, label_size], -1.0, 1.0), name='W')
            b = tf.get_variable('b', [label_size])
            self.out1 = self.BiRNN(inputs, "a", embedding_size, rnn_size, seq_length)
            self.output = tf.nn.xw_plus_b(self.out1, w, b)
            self.logits = tf.nn.softmax(self.output, dim=1)

        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(self.losses)

        with tf.name_scope("accuracy"):
            self.accuracys = tf.equal(tf.arg_max(self.logits, dimension=1), tf.arg_max(self.input_y, dimension=1), name="equal")
            self.accuracy = tf.reduce_mean(tf.cast(self.accuracys, "float"), name="accuracy")

        # with tf.name_scope("ans"):
        #     self.ans = tf.subtract(tf.ones_like(self.distance), self.distance, name="temp_sim")
        #     print(self.ans)

blstm = Model(num_layers=3,
              seq_length=30,
              embedding_size=399,
              vocab_size=120,
              rnn_size=20,
              label_size=6)