import os
import pickle
import logging
from collections import Counter, defaultdict
import argparse

from tensorflow.python.platform import gfile
import numpy as np
from os.path import join as pjoin
import tensorflow as tf
from ptpython.repl import embed


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def get_minibatches(data, minibatch_size, shuffle=True):
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)

    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]


def negative_sampling(batches):
    q_sent_batch = batches[0]
    q_len_batch = batches[1]
    c_sent_batch = batches[2]
    c_len_batch = batches[3]
    #embed(globals(),locals())
    start_label_batch = batches[4][:,0]
    end_label_batch = batches[4][:,1]
    # added negative samples
    c_sent_batch = np.tile(c_sent_batch, (2,))
    c_len_batch = np.tile(c_len_batch, (2, ))

    q_sent_batch_shuffled = np.copy(q_sent_batch)
    q_len_batch_shuffled = np.copy(q_len_batch)
    q_indices = np.arange(len(q_len_batch))
    np.random.shuffle(q_indices)
    for i,x in enumerate(q_indices):
        q_len_batch_shuffled[i] = q_len_batch[x]
        q_sent_batch_shuffled[i] = q_sent_batch[x]

    q_sent_batch = np.concatenate((q_sent_batch, q_sent_batch_shuffled), axis=0)
    q_len_batch = np.concatenate((q_len_batch,q_len_batch_shuffled),axis=0)
    start_label_batch = np.tile(start_label_batch,(2,))
    end_label_batch = np.tile(end_label_batch,(2,))
    infer_label_batch = np.concatenate((np.ones(start_label_batch.shape), np.zeros(end_label_batch.shape)), axis = 0)
    return [q_sent_batch, q_len_batch, c_sent_batch, c_len_batch,start_label_batch, end_label_batch, infer_label_batch]


def minibatches(data, batch_size, shuffle=True):
    batches = [np.array(col) for col in zip(*data)]
    batches = negative_sampling(batches)
    return get_minibatches(batches, batch_size, shuffle)


def load_glove_embeddings(glove_path):
    glove = np.load(glove_path)['glove']
    logger.info("Loading glove embedding")
    logger.info("Dimension: {}".format(glove.shape[1]))
    logger.info("Vocabulary: {}" .format(glove.shape[0]))
    logger.info("dtype of glove is: %s" % type(glove))
    logger.info("dtype of glove is: %s" % type(glove[0][0]))
    glove = tf.to_float(glove)
    logger.info("glove is: " + str(glove) )
    return glove


def load_dataset(source_dir, data_mode, max_q_toss, max_c_toss, data_pfx_list=None):

    assert os.path.exists(source_dir)

    train_pfx = pjoin(source_dir, "train")
    valid_pfx = pjoin(source_dir, "val")
    dev_pfx = pjoin(source_dir, "dev")
    data_mode = "tiny"
    if data_mode=="tiny":
        max_train = 100
        max_valid = 10
        max_dev = 10

    train = []
    valid = []
    train_raw = []
    valid_raw = []

    dev = []
    dev_raw = []

    max_c_len = 0
    max_q_len = 0

    if data_pfx_list is None:
        data_pfx_list = [train_pfx, valid_pfx]
    else:
        data_pfx_list = [pjoin(source_dir, data_pfx) for data_pfx in data_pfx_list]

    for data_pfx in data_pfx_list:
        if data_pfx == train_pfx:
            data_list = train
            data_list_raw = train_raw
            if data_mode=="tiny":
                max_entry = max_train
            logger.info("")
            logger.info("Loading training data")
        if data_pfx == valid_pfx:
            data_list = valid
            data_list_raw = valid_raw
            if data_mode=="tiny":
                max_entry = max_valid
            logger.info("")
            logger.info("Loading validation data")
        if data_pfx == dev_pfx:
            data_list = dev
            data_list_raw = dev_raw
            if data_mode=="tiny":
                max_entry = max_dev
            logger.info("")
            logger.info("Loading as dev data")

        c_ids_path = data_pfx + ".ids.context"
        c_raw_path = data_pfx + ".context"
        q_ids_path = data_pfx + ".ids.question"
        q_raw_path = data_pfx + ".question"
        label_path = data_pfx + ".span"

        counter = 0
        ignore_counter = 0

        uuid_list = []
        if data_pfx == dev_pfx:
            uuid_path = data_pfx + ".uuid"
            with gfile.GFile(uuid_path, mode="rb") as uuid_file:
                for line in uuid_file:
                    uuid_list.append(line.strip())

        with gfile.GFile(q_raw_path, mode="r") as r_q_file:
            with gfile.GFile(c_raw_path, mode="r") as r_c_file:
                with gfile.GFile(q_ids_path, mode="r") as q_file:
                    with gfile.GFile(c_ids_path, mode="r") as c_file:
                        with gfile.GFile(label_path, mode="r") as l_file:
                            for line in l_file:
                                label = list(map(int,line.strip().split(" ")))
                                context = list(map(int, c_file.readline().strip().split(" ")))
                                question = list(map(int,q_file.readline().strip().split(" ")))
                                context_raw = r_c_file.readline().strip().split(" ")
                                question_raw = r_q_file.readline().strip().split(" ")

                                c_len = len(context)
                                q_len = len(question)

                                # Do not toss out, only  truncate for dev set
                                if q_len > max_q_toss:
                                    if data_pfx == dev_pfx:
                                        q_len = max_q_toss
                                        question = question[:max_q_toss]
                                    else:
                                        ignore_counter += 1
                                        continue
                                if c_len > max_c_toss:
                                    if data_pfx == dev_pfx:
                                        c_len = max_c_toss
                                        context = context[:max_c_toss]
                                    else:
                                        ignore_counter += 1
                                        continue

                                max_c_len = max(max_c_len, c_len)
                                max_q_len = max(max_q_len, q_len)

                                entry = [question, q_len, context, c_len, label]
                                data_list.append(entry)

                                raw_entry = [question_raw, context_raw]
                                data_list_raw.append(raw_entry)

                                counter += 1
                                if counter % 10000 == 0:
                                    logger.info("read %d context lines" % counter)
                                if data_mode=="tiny":
                                    if counter==max_entry:
                                        break

        logger.info("Ignored %d questions/contexts in total" % ignore_counter)
        assert counter>0, "No questions/contexts left (likely filtered out)"

        logger.info("read %d questions/contexts in total" % counter)
        logger.info("maximum question length %d" % max_q_len)
        logger.info("maximum context length %d" % max_c_len)

    dataset = {"training":train, "validation":valid, "training_raw":train_raw, "validation_raw":valid_raw, "dev":dev, "dev_raw":dev_raw, "dev_uuid":uuid_list}
    return dataset, max_q_len, max_c_len
