# -*- coding: utf-8 -*-
#/usr/bin/python2
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
        trn_files = glob.glob(os.path.join('data_thchs30', 'xmly_yangchenghao_22050', 'archived_early', 'A*', '*.trn'))
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
        files = glob.glob(os.path.join('mels', '*.npy'))
        for f in files:
            f = f.split('/')[-1]
            trn = os.path.join('data_thchs30', 'xmly_yangchenghao_22050', 'archived_early', f[:4], f[:-4] + '.trn')
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
        lines = codecs.open(hp.test_data, 'r', 'utf-8').readlines()[1:]
        sents = [text_normalize(line.split(" ", 1)[-1]).strip() + "~" for line in lines] # text normalization, ~: EOS
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
            def _load_spectrograms(fpath):
                fname = os.path.basename(fpath).decode()
                mel = "mels/{}".format(fname.replace("wav", "npy"))
                mag = "mags/{}".format(fname.replace("wav", "npy"))
                return fname, np.load(mel), np.load(mag)

            fname, mel, mag = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
        else:
            fname, mel, mag = tf.py_func(load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])  # (None, n_mels)

        # Add shape information
        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, hp.n_mels))
        mag.set_shape((None, hp.n_fft//2+1))

        # Batching
        seq_len, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                               input_length=text_length,
                                               tensors=[text, mel, mag, fname],
                                               batch_size=hp.B,
                                               bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                               num_threads=8,
                                               capacity=hp.B*4,
                                               dynamic_pad=True)

    return texts, mels, mags, fnames, num_batch
