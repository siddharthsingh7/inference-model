import os
import logging

from tensorflow.python.platform import gfile
import numpy as np
from os.path import join
import tensorflow as tf
from ptpython.repl import embed
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def get_minibatches(data, minibatch_size, dataset):
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    if dataset == 'squad':
        squad_flag = True
    else:
        squad_flag = False
    if squad_flag:
        data = negative_sampling(data, minibatch_size)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        batches = [minibatch(d, minibatch_indices) for d in data] if list_data else minibatch(data, minibatch_indices)
        yield batches

def negative_sampling(batches, minibatch_size):
    q_final_batch = []
    q_final_len_batch = []
    c_final_batch = []
    c_final_len_batch = []
    cf_final_batch = []
    cf_final_len_batch = []
    a_final_batch = []
    a_final_len_batch = []
    infer_label_batch = []
    for q, q_l, c, c_l, cf, cf_l,a, a_l in zip(*batches):
        q_final_batch.append(q)
        q_final_len_batch.append(q_l)
        c_final_batch.append(c)
        c_final_len_batch.append(c_l)
        cf_final_batch.append(cf)
        cf_final_len_batch.append(cf_l)
        a_final_batch.append(a)
        a_final_len_batch.append(a_l)
        infer_label_batch.append(1)

        for i, qq in enumerate(batches[0]):
            if qq != q:
                q_final_batch.append(qq)
                q_final_len_batch.append(batches[1][i])
                c_final_batch.append(c)
                c_final_len_batch.append(c_l)
                cf_final_batch.append(cf)
                cf_final_len_batch.append(cf_l)
                a_final_batch.append(a)
                a_final_len_batch.append(a_l)
                infer_label_batch.append(0)
                break #
    data = [q_final_batch, q_final_len_batch, c_final_batch, c_final_len_batch, cf_final_batch, cf_final_len_batch ,a_final_batch, a_final_len_batch, infer_label_batch]
    return data


def minibatch(data, minibatch_idx):
    batches = data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]
    return batches


def minibatches(data, batch_size, dataset):
    batches = [np.array(col) for col in zip(*data)]
    return get_minibatches(batches, batch_size, dataset)


def load_glove_embeddings(glove_path):
    glove = np.load(glove_path)['glove']
    logger.info("Loading glove embedding")
    logger.info("Dimension: {}".format(glove.shape[1]))
    logger.info("Vocabulary: {}" .format(glove.shape[0]))
    glove = tf.to_float(glove)
    logger.info("glove is: " + str(glove))
    return glove


def load_dataset(source_dir, data_mode, max_q_toss, max_c_toss, data_pfx_list=None):
    '''
    From Stanford Assignment 4 starter code
    '''
    assert os.path.exists(source_dir)
    train_pfx = join(source_dir, "train")
    valid_pfx = join(source_dir, "val")
    dev_pfx = join(source_dir, "dev")
    if data_mode=="tiny":
        max_train = 500
        max_valid = 20
        max_dev = 20

    train = []
    valid = []
    train_raw = []
    valid_raw = []

    dev = []
    dev_raw = []

    max_c_len = 0
    max_q_len = 0
    max_a_len = 0

    if data_pfx_list is None:
        data_pfx_list = [train_pfx, valid_pfx]
    else:
        data_pfx_list = [join(source_dir, data_pfx) for data_pfx in data_pfx_list]

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
                                try:
                                    context_plus_features = list(map(int, c_file.readline().strip().split(" ")))
                                    question = list(map(int,q_file.readline().strip().split(" ")))
                                    context_raw = r_c_file.readline().strip().split(" ")
                                    question_raw = r_q_file.readline().strip().split(" ")
                                except Exception as e:
                                    embed(globals(),locals())
                                answers = list(map(int,context_plus_features[label[0]:label[1]]))
                                answer_raw = context_raw[label[0]:label[1]]
                                c_len = int(len(context_plus_features)/4)
                                q_len = len(question)
                                a_len = len(answers)

                                context = context_plus_features[:c_len]
                                context_features = context_plus_features[c_len:]
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
                                max_a_len = max(max_a_len, a_len)
                                entry = [question, q_len, context, c_len, context_features, len(context_features), answers,a_len]
                                data_list.append(entry)

                                raw_entry = [question_raw, context_raw, answer_raw]
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
    return dataset

def load_snli_dataset(source_dir, data_size, max_sent1_len, max_sent2_len):

    train_data = join(source_dir,'mnli.train')
    dev_data = join(source_dir,'mnli.dev.matched')
    train = []
    valid = []
    with gfile.GFile(train_data, 'r') as f:
        for line in f:
            line = line.strip().replace('[', '').replace(']', '')
            tokens = line.split(',')
            label = int(tokens[-1])
            pos = tokens.index(" ';'")
            sent1 = tokens[:pos]
            sent2 = tokens[pos+1: len(tokens) - 2]
            map(int,sent1)
            map(int,sent2)
            train.append([sent1, len(sent1), sent2, len(sent2), label])

    with gfile.GFile(dev_data, 'r') as f:
        for line in f:
            line = line.strip().replace('[', '').replace(']', '')
            tokens = line.split(',')
            label = int(tokens[-1])
            pos = tokens.index(" ';'")
            sent1 = tokens[:pos]
            sent2 = tokens[pos+1: len(tokens) - 2]
            map(int,sent1)
            map(int,sent2)
            train.append([sent1, len(sent1), sent2, len(sent2), label])

    dataset = {"training":train, "validation":valid}
    return dataset, max_sent1_len, max_sent2_len
