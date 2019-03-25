# -*- coding: utf-8 -*-
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

from tqdm import tqdm

from data_load import get_batch, load_vocab
from hyperparams import Hyperparams as hp
from modules import *
from networks import TextEnc, AudioEnc, AudioDec, Attention, SSRN
import tensorflow as tf
from utils import *
import sys
import os


class Graph:
    def __init__(self, mode="train"):
        '''
        Args:
          mode: Either "train" or "synthesize".
        '''
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()

        # Set flag
        training = True if mode=="train" else False

        # Graph
        # Data Feeding
        ## texts (B, N), int32
        ## lf0s: (B, T//r, n_lf0) float32
        ## mgcs: (B, T//r, n_mgc) float32
        ## baps: (B, T//r, n_bap) float32
        n_features = hp.n_lf0 + hp.n_mgc + hp.n_bap
        if mode=="train":
            self.texts, self.lf0s, self.mgcs, self.baps, self.fnames, self.num_batch = get_batch()
            self.features = tf.concat([tf.expand_dims(self.lf0s, axis=-1), self.mgcs, self.baps], axis=-1)
            self.prev_max_attentions = tf.ones(shape=(hp.B,), dtype=tf.int32)
            self.gts = tf.convert_to_tensor(guided_attention())
        else:  # Synthesize
            self.texts = tf.placeholder(tf.int32, shape=(None, None))
            self.features = tf.placeholder(tf.float32, shape=(None, None, n_features))
            self.prev_max_attentions = tf.placeholder(tf.int32, shape=(None,))

        with tf.variable_scope("Text2Mel"):
            # Get S or decoder inputs. (B, T//r, n_features)
            self.S = tf.concat((tf.zeros_like(self.features[:, :1, :]), self.features[:, :-1, :]), 1)

            # Networks
            with tf.variable_scope("TextEnc"):
                self.K, self.V = TextEnc(self.texts, training=training)  # (N, Tx, e)

            with tf.variable_scope("AudioEnc"):
                self.Q = AudioEnc(self.S, training=training)

            with tf.variable_scope("Attention"):
                # R: (B, T/r, 2d)
                # alignments: (B, N, T/r)
                # max_attentions: (B,)
                self.R, self.alignments, self.max_attentions = Attention(self.Q, self.K, self.V,
                                                                         mononotic_attention=(not training),
                                                                         prev_max_attentions=self.prev_max_attentions)
            with tf.variable_scope("AudioDec"):
                self.Y_logits, self.Y = AudioDec(self.R, training=training) # (B, T/r, n_features)

        with tf.variable_scope("gs"):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if training:
            # Text2Mel
            # L1 loss
            self.loss_features = tf.reduce_mean(tf.abs(self.Y - self.features))

            # binary divergence loss
            self.loss_bd1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Y_logits, labels=self.features))

            # guided_attention loss
            self.A = tf.pad(self.alignments, [(0, 0), (0, hp.max_N), (0, hp.max_T)], mode="CONSTANT", constant_values=-1.)[:, :hp.max_N, :hp.max_T]
            self.attention_masks = tf.to_float(tf.not_equal(self.A, -1))
            self.loss_att = tf.reduce_sum(tf.abs(self.A * self.gts) * self.attention_masks)
            self.mask_sum = tf.reduce_sum(self.attention_masks)
            self.loss_att /= self.mask_sum

            # total loss
            self.loss = self.loss_features + self.loss_bd1 + self.loss_att

            tf.summary.scalar('train/loss_features', self.loss_features)
            tf.summary.scalar('train/loss_bd1', self.loss_bd1)
            tf.summary.scalar('train/loss_att', self.loss_att)
            tf.summary.image('train/feature_gt', tf.expand_dims(tf.transpose(self.features[:1], [0, 2, 1]), -1))
            tf.summary.image('train/feature_hat', tf.expand_dims(tf.transpose(self.Y[:1], [0, 2, 1]), -1))

            # Training Scheme
            self.lr = learning_rate_decay(hp.lr, self.global_step)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            tf.summary.scalar("lr", self.lr)

            ## gradient clipping
            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            for grad, var in self.gvs:
                grad = tf.clip_by_value(grad, -1., 1.)
                self.clipped.append((grad, var))
                self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

            # Summary
            self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    g = Graph(); print("Training Graph loaded")

    logdir = hp.logdir
    sv = tf.train.Supervisor(logdir=logdir, save_model_secs=0, global_step=g.global_step)

    with sv.managed_session() as sess:
        # Restore saved model if the user requested it, default = True
        try:
            checkpoint_state = tf.train.get_checkpoint_state(logdir)
            if (checkpoint_state and checkpoint_state.model_checkpoint_path):
                print('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
                sv.saver.restore(sess, checkpoint_state.model_checkpoint_path)
            else:
                print('No model to load at {}'.format(logdir))
        except tf.errors.OutOfRangeError as e:
            print('Cannot restore checkpoint: {}'.format(e))

        while 1:
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                gs, _ = sess.run([g.global_step, g.train_op])

                # Write checkpoint files at every 1k steps
                if gs % 1000 == 0:
                    sv.saver.save(sess, logdir + '/model_gs_{}'.format(str(gs // 1000).zfill(3) + "k"))

                    # plot alignment
                    alignments = sess.run(g.alignments)
                    plot_alignment(alignments[0], str(gs // 1000).zfill(3) + "k", logdir)

                # break
                if gs > hp.num_iterations: break

    print("Done")
