from __future__ import division
import numpy as np
import tensorflow as tf
import logging
import tqdm
from sklearn.metrics import classification_report,accuracy_score
from cell import biLSTM, AttentionCell,get_last_layer,padding_batch
from data_util import minibatches
from sklearn.metrics import classification_report,accuracy_score

logging.basicConfig(level=logging.INFO)

from ptpython.repl import embed


class InferModel(object):
    def __init__(self, *args):
        self.config = args[0]
        self.glove_embeddings = args[1]
        self.vocab = args[2]
        self.embedding_size = self.config.embedding_size
        self.state_size = self.config.state_size

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)

        N = self.config.batch_size
        self.q = tf.placeholder(tf.int64, [N, None], name='q')
        self.x = tf.placeholder(tf.int64, [N, None], name='x')
        self.q_mask = tf.placeholder(tf.bool, [N, None], name='q_mask')
        self.x_mask = tf.placeholder(tf.bool, [N, None], name='x_mask')
        self.a = tf.placeholder(tf.int32, [N, None], name='a')
        self.a_mask = tf.placeholder(tf.bool, [N, None], name='a_mask')
        self.y = tf.placeholder(tf.int64, [N], name='y')
        self.JX = tf.placeholder(tf.int32, shape=(), name='JX')
        self.JQ = tf.placeholder(tf.int32, shape=(), name='JQ')
        self.JA = tf.placeholder(tf.int32, shape=(), name='JA')
        self.learning_rate = self.config.learning_rate

        self.W = tf.get_variable('W',shape=(2*self.state_size, self.config.num_classes), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable('b',shape=(self.config.num_classes), dtype=tf.float32, initializer=tf.constant_initializer(0))

        self.word_emb_mat = tf.get_variable(dtype=tf.float32, initializer=self.glove_embeddings, name="word_emb_mat")

        with tf.variable_scope("infer"):
            question, context, answer = self.setup_embeddings()
            # [N,JQ,2d] , [N,JX,2d], [N,JA,2d]
            with tf.variable_scope("encoder"):
                self.question_repr, self.context_repr, self.answer_repr = self.encode(question, context,
                                                                                      answer, self.x_mask,
                                                                                      self.q_mask, self.a_mask)

            with tf.variable_scope("attention_layer"):
                state_fw,state_bw = tf.split(self.context_repr, 2, axis=2)
                attention_cell_fw = AttentionCell(state_size=self.state_size, state_is_tuple=True, encoder_input=state_fw,
                                                  encoder_input_size=self.JX, encoder_mask=self.x_mask)
                attention_cell_bw = AttentionCell(state_size=self.state_size, state_is_tuple=True,
                                                  encoder_input=state_bw, encoder_input_size=self.JX, encoder_mask=self.x_mask)
 
                if self.config.dataset == 'squad':
                    attend_on = tf.concat([question, answer], 1)  # [N,JA+JQ,2*d]
                    attend_on_length_mask = tf.concat([self.q_mask, self.a_mask], 1)  #[N,JA+JQ]
                else:
                    attend_on = question  # [N,JQ,2*d]
                    attend_on_length_mask = self.q_mask  #[N,JQ]
 
                self.attended_repr, _, _ = biLSTM(attend_on, attend_on_length_mask,
                                                  cell_fw=attention_cell_fw, cell_bw=attention_cell_bw,
                                                  dropout=self.config.dropout, state_size=self.config.state_size)  #[N,JQ,2*d]



            self.decode_repr = self.decode(self.attended_repr, attend_on_length_mask)

            q_len = tf.reshape(tf.reduce_sum(tf.cast(attend_on_length_mask, tf.int32), axis=1), [-1, ])
            self.preds = get_last_layer(self.decode_repr, q_len-1)


            with tf.variable_scope("MLP_layer"):
                self.logits = tf.contrib.layers.fully_connected(self.preds, self.config.num_classes, activation_fn=None)

            self.prediction = tf.nn.softmax(self.logits)

            self.loss = self.setup_loss(self.logits, attend_on_length_mask)

            tf.summary.histogram('logits', self.logits)
            self.prediction = tf.argmax(self.prediction, 1)
            tf.summary.histogram('prediction', self.prediction)

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.y), tf.float32))

            tf.summary.scalar('accuracy', self.accuracy)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.loss)
            grads = tf.gradients(self.loss, tf.trainable_variables())
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name.replace(':', '_'), var)

            for grad in grads:
                pass

            self.inc_step = self.global_step.assign_add(1)
            self.summary_op = tf.summary.merge_all()

    def setup_loss(self, logits, mask):
        onehot_labels = tf.one_hot(self.y, 2)
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels))
        #l2_cost = 0.01 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        loss = tf.reduce_mean(tf.losses.hinge_loss(logits=logits, labels=onehot_labels))
        #loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=onehot_labels))
        #loss = loss + l2_cost
        tf.summary.scalar('loss', loss)

        return loss

    def decode(self, input, input_mask):
        """
        Couple of bilstm layers stacked
        """
        with tf.variable_scope("decoder_layer1") as scope:
            layer1, _, _ = biLSTM(input, input_mask, self.config.state_size, dropout=self.config.dropout, scope=scope)

        with tf.variable_scope("decoder_layer2") as scope:
            layer2, _, _ = biLSTM(layer1, input_mask, self.config.state_size, dropout=self.config.dropout, scope=scope)
        return layer2

    def encode(self, question, context, answer, context_mask, question_mask, answer_mask):
        """
        Returns encoded representation of inputs
        """
        with tf.variable_scope("ques_encode") as scope:
            print('-'*5 + "encoding question" + '-'*5)
            question_repr, _, _ = biLSTM(question, question_mask, dropout=self.config.dropout, state_size=self.config.state_size, scope=scope)
            print("questionrepr", question_repr)
        with tf.variable_scope("context_encode") as scope:
            print('-'*5 + "encoding context" + '-'*5)
            context_repr, _, _ = biLSTM(context, context_mask, dropout=self.config.dropout, state_size=self.config.state_size, scope=scope)
            print("context repr", context_repr)
        with tf.variable_scope("answer_encode") as scope:
            print('-'*5 + "encoding answer" + '-'*5)
            answer_repr, _, _ = biLSTM(answer, answer_mask, dropout=self.config.dropout, state_size=self.config.state_size, scope=scope)
            print("answer repr", answer_repr)

        return question_repr, context_repr, answer_repr

    def add_features(self, question, context, answer, question_mask, context_mask, answer_mask):
        """
        Exact match - binary features original,lemma or lowercase,
        Token features - POS, NER, Term Frequency
        """
        pass
    def setup_embeddings(self):
        """
        return glove pretrained embeddings for inputs
        TODO - finetune only top 1000 embedding
        """
        with tf.variable_scope("emb"), tf.device("/cpu:0"):
            question = tf.nn.embedding_lookup(self.word_emb_mat, self.q)
            context = tf.nn.embedding_lookup(self.word_emb_mat, self.x)
            answer = tf.nn.embedding_lookup(self.word_emb_mat, self.a)
            question = tf.reshape(question, [self.config.batch_size, self.JQ, self.embedding_size])
            context = tf.reshape(context, [self.config.batch_size, self.JX, self.embedding_size])
            answer = tf.reshape(answer, [self.config.batch_size, self.JA, self.embedding_size])
            return question, context, answer

    def create_feed_dict(self, question_batch, question_len_batch, context_batch, context_len_batch,
                         ans_batch, ans_len_batch, label_batch=None):

        feed_dict = {}
        JQ = np.max(question_len_batch)
        JX = np.max(context_len_batch)
        JA = np.max(ans_len_batch)

        question, question_mask = padding_batch(question_batch, JQ)
        context, context_mask = padding_batch(context_batch, JX)
        answer, answer_mask = padding_batch(ans_batch, JA)

        feed_dict[self.q] = question
        feed_dict[self.q_mask] = question_mask
        feed_dict[self.x] = context
        feed_dict[self.x_mask] = context_mask
        feed_dict[self.a] = answer
        feed_dict[self.a_mask] = answer_mask
        feed_dict[self.JQ] = JQ
        feed_dict[self.JX] = JX
        feed_dict[self.JA] = JA
        if label_batch is not None:
            feed_dict[self.y] = label_batch
        return feed_dict

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy

    def get_summary(self):
        return self.summary_op

    def get_global_step(self):
        return self.global_step
