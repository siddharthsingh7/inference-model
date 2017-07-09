from __future__ import division
import numpy as np
import tensorflow as tf
import logging
import tqdm
from sklearn.metrics import classification_report,accuracy_score
from data_util import minibatches

logging.basicConfig(level=logging.INFO)

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


def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res

class Encoder(object):
    def __init__(self,size):
        self.size = size

    def encode(self, inputs, mask, encoder_state_input):

        cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.size, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.size, state_is_tuple=True)
        #TODO add dropout

        if encoder_state_input is not None:
            state_fw = encoder_state_input[0]
            state_bw = encoder_state_input[1]
        else:
            state_fw = None
            state_bw = None
        seq_len = tf.reduce_sum(tf.cast(mask, 'int32'), axis=1)
        seq_len = tf.reshape(seq_len, [-1,])
        (hidden_state_fw,hidden_state_bw),(final_state_fw,final_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs,seq_len,state_fw,state_bw,tf.float32)

        concat_hidden_states = tf.concat([hidden_state_fw,hidden_state_bw],2)

        concat_final_state = tf.concat([final_state_fw[1], final_state_bw[1]],1)
        return concat_hidden_states,concat_final_state,(final_state_fw,final_state_bw)

class Decoder(object):
    def __init__(self,size):
        self.size = size

    def decode(self,input_repr,x_mask,q_mask,a_mask):
        seq_mask = tf.concat( [q_mask,x_mask,a_mask],1)
        with tf.variable_scope('decode_layer1'):
            print('-'*5 + "decoding layer 1" + '-'*5)
            m,_,_ = self.decode_LSTM(input_repr,seq_mask,None)
            print("first_layer",m)
        with tf.variable_scope('decode_layer2'):
            print('-'*5 + "decoding layer 2" + '-'*5)
            b,_,_ = self.decode_LSTM(m,seq_mask,None)
            print("Second_layer",b)
        return b

    def decode_LSTM(self,inputs,mask,decoder_state_input):
        cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.size,state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.size,state_is_tuple=True)

        if decoder_state_input is not None:
            state_fw = decoder_state_input[0]
            state_bw = decoder_state_input[1]
        else:
            state_fw = None
            state_bw = None
        seq_len = tf.reduce_sum(tf.cast(mask, 'int32'), axis=1)
        seq_len = tf.reshape(seq_len, [-1,])

        (hidden_state_fw,hidden_state_bw),(final_state_fw,final_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs,seq_len,state_fw,state_bw,tf.float32)

        concat_hidden_states = tf.concat([hidden_state_fw,hidden_state_bw],2)

        concat_final_state = tf.concat([final_state_fw[1], final_state_bw[1]],1)
        return concat_hidden_states,concat_final_state,(final_state_fw,final_state_bw)

class InferModel(object):
    def __init__(self, *args):
        self.config = args[0]
        self.pretrained_embeddings = args[1]
        self.vocab = args[2]
        self.embedding_size = self.config.embedding_size
        self.state_size = self.config.state_size
        self.encoder = Encoder(self.state_size)
        self.decoder = Decoder(self.state_size)

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)

        N,V= self.config.batch_size,self.vocab
        self.config.num_classes = 2
        self.q = tf.placeholder(tf.int64, [N, None],name='q')
        self.x = tf.placeholder(tf.int64, [N, None],name='x')
        self.q_mask = tf.placeholder(tf.bool,[N, None],name='q_mask')
        self.x_mask = tf.placeholder(tf.bool,[N, None],name='x_mask')
        self.a = tf.placeholder(tf.int32,[N, None],name='a')
        self.a_mask = tf.placeholder(tf.bool,[N, None],name='a_mask')
        self.y = tf.placeholder(tf.int32,[N, self.config.num_classes],name='y')
        self.JX = tf.placeholder(tf.int32,shape=(),name='JX')
        self.JQ = tf.placeholder(tf.int32,shape=(),name='JQ')
        self.JA = tf.placeholder(tf.int32,shape=(),name='JA')

        with tf.variable_scope("infer"):
            question,context,answer = self.setup_embeddings()
            print("question",question)
            print("context",context)
            self.question_repr,self.context_repr,self.answer_repr = self.encode(question,context,answer,self.x_mask,self.q_mask,self.a_mask) #[N,JQ,2d] , [N,JX,2d], [N,JA,2d]

            #match_repr = self.attention_layer(self.context_repr, self.question_repr)
            #print("match_repr",match_repr)
            self.input_repr = tf.concat( [self.question_repr,self.context_repr,self.answer_repr],1)
            print("input",self.input_repr) # [N,JX+JQ,2*d]


            self.decode_repr = self.decoder.decode(self.input_repr,self.x_mask,self.q_mask,self.a_mask)
            print("decode_repr",self.decode_repr)

            sequence_length = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), axis=1)
            self.preds = extract_axis_1(self.decode_repr,sequence_length -1)
            print("preds",self.preds)
            print('-'*5 + "SOFTMAX LAYER" + '-'*5)
            with tf.variable_scope('softmax'):
                W = tf.get_variable('W',shape=(2*self.state_size,self.config.num_classes),dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b',shape=(1),dtype=tf.float32,initializer=tf.constant_initializer(0))
                self.pred = tf.nn.softmax(tf.matmul(self.preds, W) + b)

            self.prediction = tf.argmax(self.pred,1)
            self.true_label = tf.argmax(self.y,1)
            correct_prediction = tf.equal(self.prediction,self.true_label)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
            print("preds",self.pred)
            tf.summary.histogram('logit_label', self.pred)

            self.loss = self.setup_loss(self.pred)
            # TODO decay learning rate
            # TODO dropout
            # TODO gradient clipping
            self.learning_rate = 0.0001
            self.max_gradient_norm = 0.2

            opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(self.loss,global_step=self.global_step)
            self.train_op = opt
            self.summary_op = tf.summary.merge_all()

    def attention_layer(self,context_repr,question_repr):
        with tf.variable_scope("attention layer"):
            #attention based on the paper: Reasoning about Entailment with Attention
            W_y = tf.get_variable("W_y",shape=(2*self.state_size,2*self.state_size),dtype=tf.float32)
            W_h = tf.get_variable("W_h",shape=(2*self.state_size,2*self.state_size),dtype=tf.float32)
            W_p = tf.get_variable("W_p",shape=(2*self.state_size,2*self.state_size),dtype=tf.float32)
            W_x = tf.get_variable("W_p",shape=(2*self.state_size,2*self.state_size),dtype=tf.float32)
            w = tf.get_variable("w",shape=(2*self.state_size,2*self.state_size),dtype=tf.float32)

            #TODO

    def encode(self,question,context,answer,context_mask,question_mask,answer_mask):

        with tf.variable_scope("ques_encode"):
            print('-'*5 + "encoding question" + '-'*5)
            question_repr,_,_ = self.encoder.encode(question,question_mask,encoder_state_input=None)
            print("questionrepr",question_repr)
        with tf.variable_scope("context_encode"):
            print('-'*5 + "encoding context" + '-'*5)
            context_repr,_,_ = self.encoder.encode(context,context_mask,encoder_state_input=None)
            print("context repr",context_repr)
        with tf.variable_scope("answer_encode"):
            print('-'*5 + "encoding answer" + '-'*5)
            answer_repr,_,_ = self.encoder.encode(answer,answer_mask,encoder_state_input=None)
            print("answer repr",answer_repr)

        return question_repr,context_repr,answer_repr

    def setup_embeddings(self):
        with tf.variable_scope("emb"), tf.device("/cpu:0"):
            word_emb_mat = tf.get_variable(dtype=tf.float32 ,initializer=self.pretrained_embeddings,name="word_emb_mat")
            question = tf.nn.embedding_lookup(word_emb_mat,self.q)
            context = tf.nn.embedding_lookup(word_emb_mat,self.x)
            answer = tf.nn.embedding_lookup(word_emb_mat,self.a)
            question = tf.reshape(question,[self.config.batch_size,self.JQ,self.embedding_size])
            context = tf.reshape(context,[self.config.batch_size,self.JX,self.embedding_size])
            answer = tf.reshape(answer,[self.config.batch_size,self.JA,self.embedding_size])
            return question,context,answer

    def setup_loss(self,preds):
        #loss = tf.losses.hinge_loss(logits=preds,labels=self.true_label)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=preds,labels=self.y))
        print("loss",loss)
        tf.summary.scalar('loss', loss)
        return loss

    def create_feed_dict(self, question_batch,question_len_batch,
                         context_batch, context_len_batch,
                         ans_batch,ans_len_batch,
                         label_batch=None):

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
            num_classes = 2 #(0,1)
            temp = np.zeros([len(label_batch),num_classes])
            for i,x in enumerate(label_batch):
                if x == 1:
                    temp[i][0] = 1
                else:
                    temp[i][1] = 1

            feed_dict[self.y] = temp
        return feed_dict

    def train_on_batch(self,sess,q_batch,q_len_batch,
                       c_batch,c_len_batch,a_batch,
                       a_len_batch,infer_label_batch):
        feed = self.create_feed_dict(q_batch, q_len_batch, c_batch, c_len_batch,
                                     a_batch, a_len_batch\
                                     ,label_batch=infer_label_batch)

        loss,global_step,summary = sess.run([self.loss,self.global_step,self.summary_op], feed_dict=feed)

        return loss,summary

    def run_epoch(self, sess, train_set, valid_set,train_raw,valid_raw, epoch):
        train_minibatch = minibatches(train_set, self.config.batch_size)
        global_loss = 0
        global_accuracy = 0
        set_num = len(train_set)
        batch_size = self.config.batch_size
        batch_count = int(np.ceil(set_num * 1.0 / batch_size))
        for i, batch in enumerate(train_minibatch):
            loss,summary = self.train_on_batch(sess,*batch)
            self.writer.add_summary(summary, epoch * batch_count + i)
            print("Loss-",loss)
            logging.info('-' + "EVALUATING ON TRAINING" + '-')
            train_dataset=[train_set,train_raw]
            train_score = self.evaluate_answer(sess,train_dataset)
            print("training-accuracy",train_score)
            logging.info('-' + "EVALUATING ON VALIDATION" + '-')
            valid_dataset=[train_set,train_raw]
            score = self.evaluate_answer(sess,valid_dataset)
            print("validation-accuracy",score)
            global_loss += loss
        return global_loss,summary

    def answer(self,session,test_batch):
        q_batch,q_len_batch,c_batch,c_len_batch,a_batch,a_len_batch,infer_label_answers = test_batch

        feed = self.create_feed_dict(q_batch, q_len_batch, c_batch, c_len_batch,
                                     a_batch, a_len_batch)
        output_feed = self.prediction #already argmaxed
        outputs = session.run(output_feed,feed)
        return (outputs,infer_label_answers)

    def predict_on_batch(self,session,dataset):
        predict_minibatch = minibatches(dataset,self.config.batch_size) #TODO - shuffle
        preds = []
        for i,batch in enumerate(predict_minibatch):
            preds.append(self.answer(session,batch))
        return preds

    def evaluate_answer(self,session,eval_dataset):

        batch_num = int(np.ceil(len(eval_dataset) * 1.0 / self.config.batch_size))
        eval_data = eval_dataset[0]
        eval_raw = eval_dataset[1]
        preds = self.predict_on_batch(session,eval_data)
        accuracy = 0
        for batch in preds:
            pred,true = batch
            accuracy +=  accuracy_score(true,pred)
            #print(classification_report(true, pred, target_names=['0','1']))
        accuracy = accuracy/len(preds)
        return accuracy

    def validate(self,session,dataset):
        batch_num = int(np.ceil(len(dataset) * 1.0 / self.config.batch_size))
        valid_minibatch = minibatches(dataset,self.config.batch_size)
        valid_loss = 0
        for i,batch in enumerate(valid_minibatch):
            loss = self.test(session,batch)
            valid_loss += loss
        valid_loss = valid_loss/batch_num
        return valid_loss

    def test(self,session,test_batch):
        q_batch,q_len_batch,c_batch,c_len_batch,a_batch,a_len_batch,infer_label_answers = test_set

        output_feed = [self.loss,self.prediction]
        output_loss,output_prediction = session.run(output_feed,feed)
        return output_loss


    def train(self, session, dataset):
        params = tf.trainable_variables()
        train_set = dataset['training']
        valid_set = dataset['validation']
        train_raw = dataset['training_raw']
        valid_raw = dataset['validation_raw']
        self.writer = tf.summary.FileWriter('./tmp/tensorflow', graph=tf.get_default_graph())

        for epoch in range(5):
            logging.info('-'*5 + "TRAINING-EPOCH-" + str(epoch)+ '-'*5)
            score,summary = self.run_epoch(session, train_set,valid_set,train_raw,valid_raw,epoch)
            logging.info('-'*5 + "VALIDATION" + '-'*5)
            validation_loss = self.validate(session, valid_set)
            print("validation loss",str(validation_loss))
            valid_dataset = [valid_set,valid_raw] 
            score = self.evaluate_answer(session, valid_dataset)
            print("Validation score",score)
            #TODO save the model

