import os
import json

import math
import numpy as np
import tensorflow as tf

from os.path import join
from tqdm import tqdm
from data_util import load_glove_embeddings, load_dataset
from cell import padding_batch
from infer_model import InferModel
import logging
logging.basicConfig(level=logging.INFO)
from data_util import minibatches
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.python.platform import gfile

from ptpython.repl import embed

tf.app.flags.DEFINE_string("data_dir", "./data/squad_features", "SQUAD data directory")
tf.app.flags.DEFINE_string("data_size", "tiny", "tiny/full")
tf.app.flags.DEFINE_float("learning_rate", 0.0015, "Initial learning rate ")
tf.app.flags.DEFINE_integer("num_epochs_per_decay", 6, "Epochs before reducing learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.1, "Decay factor")
tf.app.flags.DEFINE_float("max_gradient_norm", 0.0, "Norm for clipping gradients ")
tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("state_size", 100, "State Size")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Embedding Size")
tf.app.flags.DEFINE_integer("max_question_length", 60, "Maximum Question Length")
tf.app.flags.DEFINE_integer("max_context_length", 300, "Maximum Context Length")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size")
tf.app.flags.DEFINE_integer("gpu_id", 0, "gpu id")
tf.app.flags.DEFINE_float("gpu_fraction", 0.5, " % of GPU memory used.")
tf.app.flags.DEFINE_integer("num_classes", 2, "number of classes")
tf.app.flags.DEFINE_string("dataset", "squad", "squad/dontknow")
tf.app.flags.DEFINE_string("mode", "train", "train/test")
FLAGS = tf.app.flags.FLAGS


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab


class Trainer():
    def __init__(self, model, flags):
        self.model = model
        self.config = flags
        self.global_step = model.get_global_step()
        self.loss = model.get_loss()
        self.accuracy = model.get_accuracy()
        self.summary_op = model.get_summary()
        self.train_writer = tf.summary.FileWriter('./temp/train/', graph=tf.get_default_graph())
        self.valid_writer = tf.summary.FileWriter('./temp/valid/', graph=tf.get_default_graph())

    def run_epoch(self, session, train_set, train_raw,epoch):
        total_batches = int(len(train_set) / self.config.batch_size)
        train_minibatches = minibatches(train_set, self.config.batch_size, self.config.dataset)
        training_loss = 0.0
        training_accuracy = 0.0
        for batch in tqdm(train_minibatches, desc="Trainings", total=total_batches):
            if len(batch[0]) != self.config.batch_size:
                continue
            session.run(self.model.inc_step)
            #TODO Learning rate decay
            loss, accuracy, summary, global_step = self.train_single_batch(session,*batch)
            self.train_writer.add_summary(summary, global_step)
            training_accuracy += accuracy
            training_loss += loss
        training_loss = training_loss/total_batches
        training_accuracy = training_accuracy/total_batches
        print("Loss", training_loss)
        print("Accuracy", training_accuracy)


    def train_single_batch(self,session,*batch):
        q_batch,q_len_batch,c_batch,c_len_batch,a_batch,a_len_batch,infer_label_batch = batch
        input_feed  = self.model.create_feed_dict(q_batch,q_len_batch,c_batch,c_len_batch,
                                            a_batch,a_len_batch,label_batch=infer_label_batch)
        output_feed = [self.model.train_op,self.loss,self.global_step,self.accuracy,self.summary_op]
        _,loss,global_step,accuracy,summary = session.run(output_feed,feed_dict=input_feed)
        return loss,accuracy,summary,global_step

    def validate(self,session,validation_set,validation_raw,epoch):
        total_batches = int(len(validation_set)/self.config.batch_size)
        validation_accuracy = 0.0
        validation_loss = 0.0
        validate_minibatches = minibatches(validation_set,self.config.batch_size,self.config.dataset)
        for batch in tqdm(validate_minibatches,total=total_batches,desc="Validate"):
            if len(batch[0]) != self.config.batch_size:
                continue
            loss,accuracy,summary,global_step = self.validate_single_batch(session,*batch)
            self.valid_writer.add_summary(summary, global_step)
            validation_accuracy += accuracy
            validation_loss += loss
        validation_loss = validation_loss/total_batches
        validation_accuracy = validation_accuracy/total_batches
        print("Loss",validation_loss)
        print("Accuracy",validation_accuracy)

    def validate_single_batch(self,session,*batch):
        q_batch,q_len_batch,c_batch,c_len_batch,a_batch,a_len_batch,infer_label_batch = batch
        input_feed  = self.model.create_feed_dict(q_batch,q_len_batch,c_batch,c_len_batch,
                                            a_batch,a_len_batch,label_batch=infer_label_batch)
        output_feed = [self.loss,self.global_step,self.accuracy,self.summary_op,self.model.prediction]
        loss,global_step,accuracy,summary_op,prediction = session.run(output_feed,feed_dict=input_feed)
        #print(classification_report(infer_label_batch, prediction, target_names=['0','1']))
        #print("prediction",prediction)
        #print("true",infer_label_batch)
        return loss, accuracy, summary_op, global_step

    def predict_single_batch(self,session,*batch):
        pass

    def predict_single(self,session,*batch):
        pass

class Tester():
    def __init__(self):
        pass


def load_dontknow_dataset(data_size,max_question_length,max_context_length):
    train_data = join("data","dont_know","dn_features.train")
    dev_data = join("data","dont_know","dn_features.dev")
    train = []
    valid = []
    def convert2int(x):
        if x == ' ' or x == '':
            pass
        else:
            return(int(x))

    with gfile.GFile(train_data, 'r') as f:
        for line in f:
            line = line.strip().replace('[', '').replace(']', '')
            tokens = line.split(", ';;;',") # Hacky.. #TODO fix the preprocessing script
            data = [list(map(int, x.strip().split(','))) for x in tokens]
            #   data = sent1,sent2,pos1,pos2,sim_word,sim_lemma,sim_lower,label
            sent1 = data[0] + data[2] + data[4] + data[5] + data[6] + data[7]# sent1 + pos1 +  + sim_word + sim_lemma + sim_lower
            sent2 = data[1] + data[3]
            label = data[-1]
            train.append([sent1, len(sent1), sent2, len(sent2), [], 0, label[0]])

    with gfile.GFile(dev_data, 'r') as f:
        for line in f:
            line = line.strip().replace('[', '').replace(']', '')
            tokens = line.split(", ';;;',") # Hacky.. #TODO fix the preprocessing script
            data = [list(map(int, x.strip().split(','))) for x in tokens]
         #   data = sent1,sent2,pos1,pos2,sim_word,sim_lemma,sim_lower,label
            sent1 = data[0] + data[2] + data[4] + data[5] + data[6] + data[7]# sent1 + pos1 +  + sim_word + sim_lemma + sim_lower
            sent2 = data[1] + data[3]
            label = data[-1]
            valid.append([sent1, len(sent1), sent2, len(sent2), [], 0, label[0]])

    if data_size=="tiny":
        train = train[:100]
        valid = valid[:10]

    dataset = {"training":train, "validation":valid,"training_raw":[],"validation_raw":[]}
    return dataset

def train():
    if FLAGS.dataset == 'dontknow':
        dataset = load_dontknow_dataset(FLAGS.data_size,FLAGS.max_question_length,FLAGS.max_context_length)
        embed_path = join("data","dont_know","glove.trimmed.100.npz")
        vocab_path = join("data","dont_know", "vocab.dat")
        vocab, rev_vocab = initialize_vocab(vocab_path)
        embeddings = load_glove_embeddings(embed_path)

    else:
        dataset = load_dataset(FLAGS.data_dir,FLAGS.data_size,FLAGS.max_question_length,FLAGS.max_context_length)
        embed_path = join("data", "squad", "glove.trimmed.100.npz")
        vocab_path = join(FLAGS.data_dir, "vocab.dat")
        vocab, rev_vocab = initialize_vocab(vocab_path)
        embeddings = load_glove_embeddings(embed_path)

    model = InferModel(FLAGS, embeddings, vocab)

    trainer = Trainer(model,FLAGS)

    with tf.device("/gpu:{}".format(FLAGS.gpu_id)):
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
        with tf.Session(config=config) as sess:
            logging.info("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
            logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
            train_set = dataset['training']
            valid_set = dataset['validation']
            train_raw = dataset['training_raw']
            valid_raw = dataset['validation_raw']

            for epoch in range(FLAGS.num_epochs):
                #TODO decay learning rate
                logging.info('-'*5 + "TRAINING-EPOCH-" + str(epoch)+ '-'*5)
                trainer.run_epoch(sess,train_set,train_raw,epoch)
                logging.info('-'*5 + "-VALIDATE-" + str(epoch)+ '-'*5)
                trainer.validate(sess,valid_set,valid_raw,epoch)

def test():
    pass

def main(_):
    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'test':
        test()

if __name__ == "__main__":
    tf.app.run()
