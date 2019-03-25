# -*- coding: utf-8 -*-
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import glob
import re
import os
import unicodedata

def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char

def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_data(mode="train"):
    '''Loads data
      Args:
          mode: "train" or "synthesize".
    '''
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    if mode=="prepro":
        # Parse
        fpaths, text_lengths, texts = [], [], []
        trn_files = glob.glob(os.path.join('data_thchs30', 'xmly_yangchenghao_16000', 'archived_good', 'A*', '*.trn'))
        # trn_files = glob.glob(os.path.join('data_thchs30', 'biaobei_48000', '*.trn'))
        for trn in trn_files:
            with open(trn) as f:
                fpath = trn[:-4] + '.wav'
                fpaths.append(fpath)
                text = f.readline().strip()
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32).tostring())
        return fpaths, text_lengths, texts
    elif mode=="train":
        fpaths, text_lengths, texts = [], [], []
        files = glob.glob(os.path.join('lf0', '*.npy'))
        for f in files:
            f = f.split('/')[-1]
            trn = os.path.join('data_thchs30', 'xmly_yangchenghao_16000', 'archived_good', f[:4], f[:-4] + '.trn')
            with open(trn) as trn_f:
                fpath = trn[:-4] + '.wav'
                fpaths.append(fpath)
                text = trn_f.readline().strip()
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32).tostring())
        return fpaths, text_lengths, texts
    else: # synthesize on unseen test text.
        # Parse
        # lines = codecs.open(hp.test_data, 'r', 'utf-8').readlines()[1:]
        # sents = [text_normalize(line.split(" ", 1)[-1]).strip() + "~" for line in lines] # text normalization, ~: EOS
        sents = hp.sentences
        texts = np.zeros((len(sents), hp.max_N), np.int32)
        for i, sent in enumerate(sents):
            texts[i, :len(sent)] = [char2idx[char] for char in sent]
        return texts

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        fpaths, text_lengths, texts = load_data('train') # list
        maxlen, minlen = max(text_lengths), min(text_lengths)

        # Calc total batch count
        num_batch = len(fpaths) // hp.B

        # Create Queues
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)

        # Parse
        text = tf.decode_raw(text, tf.int32)  # (None,)

        if hp.prepro:
            def _load_features(fpath):
                fname = os.path.basename(fpath).decode()
                lf0 = "lf0/{}".format(fname.replace("wav", "npy"))
                mgc = "mgc/{}".format(fname.replace("wav", "npy"))
                bap = "bap/{}".format(fname.replace("wav", "npy"))
                return fname, np.load(lf0), np.load(mgc), np.load(bap)

            fname, lf0, mgc, bap = tf.py_func(_load_features, [fpath], [tf.string, tf.float32, tf.float32, tf.float32])
        else:
            fname, lf0, mgc, bap = tf.py_func(load_features, [fpath], [tf.string, tf.float32, tf.float32, tf.float32])

        # Add shape information
        fname.set_shape(())
        text.set_shape((None,))
        lf0.set_shape((None,))
        mgc.set_shape((None, hp.n_mgc))
        bap.set_shape((None, hp.n_bap))

        # Batching
        seq_len, (texts, lf0s, mgcs, baps, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                                     input_length=text_length,
                                                     tensors=[text, lf0, mgc, bap, fname],
                                                     batch_size=hp.B,
                                                     bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                                     num_threads=8,
                                                     capacity=hp.B*4,
                                                     dynamic_pad=True)

    return texts, lf0s, mgcs, baps, fnames, num_batch
