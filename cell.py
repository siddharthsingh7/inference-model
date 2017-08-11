import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper, RNNCell, LSTMStateTuple
import numpy as np
from ptpython.repl import embed

def add_paddings(sentence, max_length):
    mask = [True] * len(sentence)
    pad_len = max_length - len(sentence)
    if pad_len > 0:
        padded_sentence = sentence + [0] * pad_len
        mask += [False] * pad_len
    else:
        padded_sentence = sentence[:max_length]
        mask = mask[:max_length]
    return padded_sentence, mask


def padding_batch(data, max_len):
    padded_data = []
    padded_mask = []
    for sentence in data:
        d, m = add_paddings(sentence, max_len)
        padded_data.append(d)
        padded_mask.append(m)
    return (padded_data, padded_mask)


def get_last_layer(data, ind):
    """
    https://stackoverflow.com/a/43298689
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res


def biLSTM(inputs, mask, state_size, cell_fw=None,cell_bw=None,dropout=None,scope=None):
    if scope is None:
        scope = "biLSTM"
    with tf.variable_scope(scope):

        if cell_fw:
            cell_fw = cell_fw
            cell_bw = cell_bw
        else:
            cell_fw = tf.contrib.rnn.LSTMCell(num_units=state_size, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(num_units=state_size, state_is_tuple=True)

        if dropout:
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=dropout)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=dropout)

        seq_len = tf.reshape(tf.reduce_sum(tf.cast(mask, 'int32'), axis=1), [-1, ])

        (hidden_state_fw, hidden_state_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, seq_len, dtype=tf.float32)

        concat_hidden_states = tf.concat([hidden_state_fw, hidden_state_bw], 2)
        concat_final_state = tf.concat([final_state_fw[1], final_state_bw[1]], 1)

        return concat_hidden_states, concat_final_state, (final_state_fw, final_state_bw)


class TreeLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    '''
    Child Sum LSTM Tree - For dependency trees
    '''
    def __init__(self,state_size,state_is_tuple):
        self._keep_prob = 0.8
        super(TreeLSTMCell,self).__init__(state_size,state_is_tuple)

    def __call__(self,inputs,state):
        #TODO
        lhs, rhs = state
        c0, h0 = lhs
        c1, h1 = rhs
        concat = tf.contrib.layers.linear(
            tf.concat([inputs, h0, h1], 1), 5 * self._num_units)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f0, f1, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

        j = self._activation(j)
        if not isinstance(self._keep_prob, float) or self._keep_prob < 1:
            j = tf.nn.dropout(j, self._keep_prob)

        new_c = (c0 * tf.sigmoid(f0 + self._forget_bias) +
                 c1 * tf.sigmoid(f1 + self._forget_bias) +
                 tf.sigmoid(i) * j)
        new_h = self._activation(new_c) * tf.sigmoid(o)

        new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        return new_h, new_state

def softmax_masked(scores, mask):
    exp = tf.exp(scores - tf.reduce_max(scores, 1, keep_dims=True)) * mask
    return tf.div(exp, tf.reduce_sum(exp, 1, keep_dims=True))

class MatchLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    '''
    MatchLSTMCell from  arXiv:1608.07905v2
    '''
    def __init__(self,state_size,state_is_tuple,encoder_input,encoder_input_size,encoder_mask):
        self.d = state_size
        self.Y = encoder_input
        self.enc_size = encoder_input_size
        self.mask = encoder_mask
        super(MatchLSTMCell,self).__init__(state_size,state_is_tuple)

    def __call__(self,inputs,state,scope=None):
        with tf.variable_scope("attention_cell"):
            c,h = state
            V,W_a,b_a,v,b = self.get_weights(self.d)
            F_part1 = tf.reshape(tf.matmul(tf.reshape(self.Y, [-1, self.d]), V), [-1, self.enc_size, self.d])
            F_part2 = tf.expand_dims(tf.matmul(h, W_a) + b_a, 1)
            F = tf.tanh(F_part1 + F_part2)
            pre_softmax_score = tf.reshape(tf.matmul(tf.reshape(F, [-1, self.d]), tf.expand_dims(v, 1)), [-1, self.enc_size]) + b
            beta = tf.nn.softmax(pre_softmax_score)
            h_beta = tf.reshape(tf.matmul(tf.expand_dims(beta, 1), self.Y), [-1, self.d])
            return (h_beta, tf.contrib.rnn.LSTMStateTuple(c,h_beta))

    @staticmethod
    def get_weights(state_size):
        xavier_init= tf.contrib.layers.xavier_initializer()
        xavier_init = tf.orthogonal_initializer()
        zero_init = tf.constant_initializer(0)
        V = tf.get_variable("V",shape=[state_size,state_size],dtype=tf.float32,initializer=xavier_init)
        W_a = tf.get_variable("W_a",shape=[state_size,state_size],dtype=tf.float32,initializer=xavier_init)
        b_a = tf.get_variable("b_a",shape=[state_size,],dtype=tf.float32,initializer=zero_init)
        v = tf.get_variable("v",shape=[state_size,],dtype=tf.float32,initializer=zero_init)
        b = tf.get_variable("b",shape=[],dtype=tf.float32,initializer=zero_init)
        return V, W_a, b_a, v, b


#TODO RECURRENT DROPOUT

class ReReadLSTM(tf.contrib.rnn.BasicLSTMCell):
    '''
    Re-Read LSTM from https://www.aclweb.org/anthology/C/C16/C16-1270.pdf
    '''
    def __init__(self,state_size,state_is_tuple,encoder_input,encoder_input_size,encoder_mask):
        self.d = state_size
        self.Y = encoder_input
        self.enc_size = encoder_input_size
        self.mask = encoder_mask
        super(ReReadLSTM,self).__init__(state_size,state_is_tuple)

    def __call__(self,inputs,state,scope=None):
        with tf.variable_scope("rereadLSTM"):
            c,h = state

        return (h,tf.contrib.rnn.LSTMStateTuple(c,h))

class AttentionCell(tf.contrib.rnn.BasicLSTMCell):
    def __init__(self,state_size,state_is_tuple,encoder_input,encoder_input_size,encoder_mask):
        self.d = state_size
        self.Y = encoder_input
        self.enc_size = encoder_input_size
        self.mask = encoder_mask
        super(AttentionCell,self).__init__(state_size,state_is_tuple)

    def __call__(self,inputs,state,scope=None):
        with tf.variable_scope("attention_cell"):
            c,h = state
            W_y,W_h,W_p,W_x,w = self.get_weights(self.d)
            m1 = tf.reshape(tf.matmul(tf.reshape(self.Y, [-1,self.d]),W_y),[-1,self.enc_size,self.d],name="m1")
            m2 = tf.expand_dims(tf.matmul(inputs, W_h) ,1,name="m2")
            M = tf.tanh(m1+m2)
            self.mask = tf.cast(self.mask,tf.float32)
            alpha = tf.reshape(tf.matmul(tf.reshape(M,[-1,self.d]),tf.expand_dims(w,1)),[-1,self.enc_size])
            print(alpha)
            alpha = softmax_masked(alpha,self.mask)
            alpha = tf.nn.softmax(alpha,name="alpha")
            r = tf.reshape(tf.matmul(tf.expand_dims(alpha,1),self.Y),[-1,self.d])
            h_star = tf.tanh( tf.matmul(r,W_p) + tf.matmul(inputs,W_x))
            return (h_star, tf.contrib.rnn.LSTMStateTuple(c,h_star))

    @staticmethod
    def get_weights(state_size):
        xavier_init= tf.contrib.layers.xavier_initializer()
        xavier_init = tf.orthogonal_initializer()
        zero_init = tf.constant_initializer(0)
        W_y = tf.get_variable("W_y",shape=[state_size,state_size],dtype=tf.float32,initializer=xavier_init)
        W_h = tf.get_variable("W_h",shape=[state_size,state_size],dtype=tf.float32,initializer=xavier_init)
        W_p = tf.get_variable("W_p",shape=[state_size,state_size],dtype=tf.float32,initializer=xavier_init)
        W_x = tf.get_variable("W_x",shape=[state_size,state_size],dtype=tf.float32,initializer=xavier_init)
        w = tf.get_variable("w",shape=[state_size,],dtype=tf.float32,initializer=xavier_init)
        return W_y,W_h,W_p,W_x,w

