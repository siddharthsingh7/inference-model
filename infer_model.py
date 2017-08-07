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
        self.q = tf.placeholder(tf.int64, [None, None], name='q')
        self.x = tf.placeholder(tf.int64, [None, None], name='x')
        self.fx = tf.placeholder(tf.int64, [None, None], name ='fx')
        self.q_mask = tf.placeholder(tf.bool, [None, None], name='q_mask')
        self.x_mask = tf.placeholder(tf.bool, [None, None], name='x_mask')
        self.fx_mask = tf.placeholder(tf.bool, [None, None], name='fx_mask')
        self.a = tf.placeholder(tf.int32, [None, None], name='a')
        self.a_mask = tf.placeholder(tf.bool, [None, None], name='a_mask')
        self.y = tf.placeholder(tf.int64, [None], name='y')
        self.JX = tf.placeholder(tf.int32, shape=(), name='JX')
        self.JQ = tf.placeholder(tf.int32, shape=(), name='JQ')
        self.JA = tf.placeholder(tf.int32, shape=(), name='JA')
        self.JFX = tf.placeholder(tf.int32, shape=(), name='JFX')
        self.learning_rate = tf.placeholder(tf.float32,shape=(),name="learning_rate")

        # Decay the learning rate exponentially based on the number of steps.
        self.lr = tf.train.exponential_decay(self.learning_rate,
                                             self.global_step,
                                             self.config.num_per_decay,
                                             self.config.decay_factor,
                                             staircase=True)
        self.W = tf.get_variable('W',shape=(2*self.state_size, self.config.num_classes), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable('b',shape=(self.config.num_classes), dtype=tf.float32, initializer=tf.constant_initializer(0))

        self.word_emb_mat = tf.get_variable(dtype=tf.float32, initializer=self.glove_embeddings, name="word_emb_mat")

        with tf.variable_scope("infer"):
            question, context, answer = self.setup_embeddings()
            # [N,JQ,2d] , [N,JX,2d], [N,JA,2d]

            with tf.variable_scope("aligned_embeddings"):
                # TODO Add RELU layer
                xx = tf.tile(tf.expand_dims(context,2),[1,1,self.JQ,1])
                yy = tf.tile(tf.expand_dims(question,1),[1,self.JX,1,1])
                alpha = tf.exp(tf.multiply(xx,yy))
                sum_matrix = tf.tile(tf.reduce_sum(alpha,2,keep_dims=True),[1,1,self.JQ,1]) # [N,JX,JQ,d]
                sum_matrix = sum_matrix - alpha
                alpha = tf.divide(alpha, sum_matrix) # [N,JX,JQ,d]
                aligned_embed = tf.reduce_sum(tf.multiply(alpha,yy),2,name="reduce_sum")
                print("Q aligned_embed",aligned_embed)# [N, JX,d]

            with tf.variable_scope("concatenate_context_embeddings"):
                context = tf.add(context,aligned_embed) # TODO concatenate, instead of add
                # Add context_features

            with tf.variable_scope("encoder"):
                self.question_repr, self.context_repr, self.answer_repr = self.encode(question, context,
                                                                                      answer, self.q_mask,
                                                                                      self.x_mask, self.a_mask)

            with tf.variable_scope("attention_layer"):
                state_fw, state_bw = tf.split(self.context_repr, 2, axis=2)
                attention_cell_fw = AttentionCell(state_size=self.state_size, state_is_tuple=True, encoder_input=state_fw,
                                                  encoder_input_size= self.JX, encoder_mask=self.x_mask)
                attention_cell_bw = AttentionCell(state_size=self.state_size, state_is_tuple=True,
                                                  encoder_input=state_bw, encoder_input_size=self.JX, encoder_mask=self.x_mask)

                if self.config.dataset == 'squadd':
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
            self.train_op = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.999).minimize(self.loss)

            grads = tf.gradients(self.loss, tf.trainable_variables())
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name.replace(':', '_'), var)

            for grad in grads: # none issues
                pass

            self.inc_step = self.global_step.assign_add(1)
            self.summary_op = tf.summary.merge_all()

    def setup_loss(self, logits, mask):
        onehot_labels = tf.one_hot(self.y, 2)
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels))
        #l2_cost = self.config.l2_beta * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        #loss = tf.reduce_mean(tf.losses.hinge_loss(logits=logits, labels=onehot_labels))
        loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=onehot_labels))
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

    def encode(self, question, context, answer, question_mask, context_mask, answer_mask):
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

    def setup_embeddings(self):
        """
        return glove pretrained embeddings for inputs
        """
        with tf.variable_scope("emb"), tf.device("/cpu:0"):
            question = tf.nn.embedding_lookup(self.word_emb_mat, self.q)
            context = tf.nn.embedding_lookup(self.word_emb_mat, self.x)
            answer = tf.nn.embedding_lookup(self.word_emb_mat, self.a)
            #question = tf.reshape(question, [self.config.batch_size, self.JQ, self.embedding_size])
            #context = tf.reshape(context, [self.config.batch_size, self.JX, self.embedding_size])
            #answer = tf.reshape(answer, [self.config.batch_size, self.JA, self.embedding_size])
            question = tf.reshape(question, [-1, self.JQ, self.embedding_size])
            context = tf.reshape(context, [-1, self.JX, self.embedding_size])
            answer = tf.reshape(answer, [-1, self.JA, self.embedding_size])
            return question, context, answer

    def create_feed_dict(self, question_batch, question_len_batch, context_batch, context_len_batch,
                         context_features_batch, context_features_len_batch, ans_batch, ans_len_batch, label_batch=None):

        feed_dict = {}
        JQ = np.max(question_len_batch)
        JX = np.max(context_len_batch)
        JA = np.max(ans_len_batch)
        JFX = np.max(context_features_len_batch)

        question, question_mask = padding_batch(question_batch, JQ)
        context, context_mask = padding_batch(context_batch, JX)
        answer, answer_mask = padding_batch(ans_batch, JA)
        context_features, context_features_mask = padding_batch(context_batch, JX)

        feed_dict[self.q] = question
        feed_dict[self.q_mask] = question_mask
        feed_dict[self.x] = context
        feed_dict[self.x_mask] = context_mask
        feed_dict[self.fx] = context_features
        feed_dict[self.fx_mask] = context_features_mask
        feed_dict[self.a] = answer
        feed_dict[self.a_mask] = answer_mask
        feed_dict[self.JQ] = JQ
        feed_dict[self.JX] = JX
        feed_dict[self.JA] = JA
        feed_dict[self.JFX] = JFX
        feed_dict[self.learning_rate] = self.config.learning_rate
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
