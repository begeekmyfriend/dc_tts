# -*- coding: utf-8 -*-
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

import os

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_data
from scipy.io.wavfile import write
from tqdm import tqdm

def synthesize():
    # Load data
    texts = load_data("synthesize")

    # Load graph
    g = Graph(mode="synthesize"); print("Graph loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-1"))
        print("Text2Mel Restored!")

        # Feed Forward
        n_features = hp.n_lf0 + hp.n_mgc + hp.n_bap
        Y = np.zeros((len(texts), hp.max_T, hp.n_features), np.float32)
        prev_max_attentions = np.zeros((len(texts),), np.int32)
        for j in tqdm(range(hp.max_T)):
            _gs, _Y, _max_attentions, _alignments = \
                sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                         {g.texts: texts,
                          g.features: Y,
                          g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]

        # Generate wav files
        if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
        for i, features in enumerate(Y):
            print("Working on file", i+1)
            lf0 = features[:, :, :hp.n_lf0]
            mgc = features[:, :, hp.n_lf0 : hp.n_mgc]
            bap = features[:, :, hp.n_lf0 + hp.n_mgc : hp.n_bap]
            wav = synthesize(mag)
            save_wav(wav, hp.sampledir + "/{:03}.wav".format(i+1))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    synthesize()
    print("Done")
