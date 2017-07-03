import os
import json


import numpy as np
import tensorflow as tf

from os.path import join as pjoin
from data_util import load_glove_embeddings, load_dataset
from model import InferModel
import logging
from ptpython.repl import embed

logging.basicConfig(level=logging.INFO)


tf.app.flags.DEFINE_string("data_dir", "./data/squad", "SQUAD data directory")
tf.app.flags.DEFINE_string("data_size", "debug", "debug/full")
tf.app.flags.DEFINE_integer("max_question_length", 60,"Maximum Question Length")
tf.app.flags.DEFINE_integer("max_context_length", 300,"Maximum Context Length")
tf.app.flags.DEFINE_integer("batch_size", 10,"batch_size")
tf.app.flags.DEFINE_integer("gpu_id", 0, "Which GPU to use.")
tf.app.flags.DEFINE_float("gpu_fraction", 0.5, "Fraction of the GPU memory to allocate.")
FLAGS = tf.app.flags.FLAGS


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def main(_):
    dataset, max_q_len, max_c_len = load_dataset(FLAGS.data_dir,
                                                 FLAGS.data_size,
                                                 FLAGS.max_question_length,
                                                 FLAGS.max_context_length)

    embed_path = pjoin("data", "squad", "glove.trimmed.100.npz")
    vocab_path = pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)
    embeddings = load_glove_embeddings(embed_path)


    num_per_epoch = len(dataset['training'])
    model = InferModel(FLAGS,embeddings,num_per_epoch,vocab)
    with tf.device("gpu:{}".format(FLAGS.gpu_id)):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
        config.allow_soft_placement = True

        with tf.Session(config=config) as sess:
            logging.info("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
            logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
            model.train(sess,dataset)

if __name__ == "__main__":
    tf.app.run()
