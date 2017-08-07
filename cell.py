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


# TODO TREELSTM

# TODO BIMPM

def softmax_masked(scores, mask):
    """
    Used to calculcate a softmax score with true sequence length (without padding), rather than max-sequence length.

    Input shape: (batch_size, max_seq_length, hidden_dim). 
    mask parameter: Tensor of shape (batch_size, max_seq_length). Such a mask is given by the length() function.
    """
    numerator = tf.exp(tf.subtract(scores, tf.reduce_max(scores, 1, keep_dims=True))) * mask
    denominator = tf.reduce_sum(numerator, 1, keep_dims=True)
    weights = tf.div(numerator, denominator)
    return weights


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
            #TODO masked softmask
            self.mask = tf.cast(self.mask,tf.float32)
            alpha = tf.reshape(tf.matmul(tf.reshape(M,[-1,self.d]),tf.expand_dims(w,1)),[-1,self.enc_size])
            print(alpha)
            alpha = softmax_masked(alpha,self.mask)
            alpha = tf.nn.softmax(alpha,name="alpha")
            r = tf.reshape(tf.matmul(tf.expand_dims(alpha,1),self.Y),[-1,self.d])
            h_star = tf.tanh( tf.matmul(r,W_p) + tf.matmul(inputs,W_x))
            return (h_star, tf.contrib.rnn.LSTMStateTuple(h_star,h_star))

    @staticmethod
    def get_weights(state_size):
        xavier_init= tf.contrib.layers.xavier_initializer()
        xavier_init = tf.orthogonal_initializer()
        zero_init = tf.constant_initializer(0)
        W_y = tf.get_variable("W_y",shape=[state_size,state_size],dtype=tf.float32,initializer=xavier_init)
        W_h = tf.get_variable("W_h",shape=[state_size,state_size],dtype=tf.float32,initializer=xavier_init)
        W_p = tf.get_variable("W_p",shape=[state_size,state_size],dtype=tf.float32,initializer=xavier_init)
        W_x = tf.get_variable("W_x",shape=[state_size,state_size],dtype=tf.float32,initializer=xavier_init)
        w = tf.get_variable("w",shape=[state_size,],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        return W_y,W_h,W_p,W_x,w

'''
class AttentionCell(tf.contrib.rnn.BasicLSTMCell):
    """
    (arXiv:1509.06664) Reasoning about Entailment with Neural Attention
    """
    def __init__(self, state_size, state_is_tuple, encoder_input, encoder_input_size):
        self._d = state_size
        self._Y = encoder_input
        self._enc_size = encoder_input_size
        super(AttentionCell, self).__init__(state_size, state_is_tuple)

    def __call__(self, inputs, state, scope):
        with tf.variable_scope("attention_cell"):
            W_y, W_h, W_p, W_x, w = self.get_weights(self.state_size)
            m1 = tf.reshape(tf.matmul(tf.reshape(self.encoder_input, [-1, self.state_size]), W_y), [-1, self.encoder_input_size,self.state_size], name="m1")
            m2 = tf.expand_dims(tf.matmul(inputs, W_h), 1, name="m2")
            M = tf.tanh(m1+m2)
            #TODO masked softmax
            alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, self.state_size]), tf.expand_dims(w, 1)), [-1, self.encoder_input_size], name="alpha"))
            r = tf.reshape(tf.matmul(tf.expand_dims(alpha, 1), self.encoder_input), [-1, self.state_size])
            h_star = tf.tanh(tf.matmul(r, W_p) + tf.matmul(inputs, W_x))
            return (h_star, state)

    @staticmethod
    def get_weights(state_size):
        xavier_init = tf.contrib.layers.xavier_initializer()

        W_y = tf.get_variable("W_y",shape=[state_size,state_size], dtype=tf.float32, initializer=xavier_init)
        W_h = tf.get_variable("W_h",shape=[state_size,state_size], dtype=tf.float32, initializer=xavier_init)
        W_p = tf.get_variable("W_p",shape=[state_size,state_size], dtype=tf.float32, initializer=xavier_init)
        W_x = tf.get_variable("W_x",shape=[state_size,state_size], dtype=tf.float32, initializer=xavier_init)
        w = tf.get_variable("w",shape=[state_size,], dtype=tf.float32, initializer=xavier_init)

        return W_y, W_h, W_p, W_x, w

    @property
    def state_size(self):
        return self._d

    @property
    def encoder_input(self):
        return self._Y

    @property
    def encoder_input_size(self):
        return self._enc_size
'''
