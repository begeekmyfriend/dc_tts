# -*- coding: utf-8 -*-
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''
from __future__ import print_function, division

import pysptk
import pyworld as vocoder
import soundfile as sf
import numpy as np
import os
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from hyperparams import Hyperparams as hp
import tensorflow as tf

int16_max = 32768.0
lf0_bias = 3
mgc_bias = 3
bap_bias = 9
bap_scale = 4

def save_wav(wav, path):
    sf.write(path, wav, hp.sr)

def synthesize(lf0, mgc, bap):
    lf0 = lf0 + lf0_bias
    mgc = mgc + mgc_bias
    bap = bap / bap_scale + bap_bias
    lf0 = np.where(lf0 < 1, 0.0, lf0)
    f0 = f0_denormalize(lf0)
    sp = sp_denormalize(mgc)
    ap = ap_denormalize(bap, lf0)
    wav = vocoder.synthesize(f0, sp, ap, hp.sr)
    return wav

def f0_norm(x):
    lf0 = np.log(np.where(x == 0.0, 1.0, x)).astype(np.float32)
    return lf0 - lf0_bias

def f0_denorm(x):
    return np.where(x == 0.0, 0.0, np.exp(x.astype(np.float64)))

def sp_norm(x):
    sp = int16_max * np.sqrt(x)
    mgc = pysptk.sptk.mcep(sp.astype(np.float32), order=hp.n_mgc - 1, alpha=hp.mcep_alpha,
                           maxiter=0, threshold=0.001, etype=1, eps=1.0E-8, min_det=0.0, itype=3)
    return mgc - mgc_bias

def sp_denorm(x):
    sp = pysptk.sptk.mgc2sp(x.astype(np.float64), order=hp.n_mgc - 1,
                            alpha=hp.mcep_alpha, gamma=0.0, fftlen=hp.n_fft)
    return np.square(sp / int16_max)

def ap_norm(x):
    ap = int16_max * np.sqrt(x)
    bap = pysptk.sptk.mcep(ap.astype(np.float32), order=hp.n_bap - 1, alpha=hp.mcep_alpha,
                           maxiter=0, threshold=0.001, etype=1, eps=1.0E-8, min_det=0.0, itype=3)
    return (bap - bap_bias) * bap_scale

def ap_denorm(x, lf0):
    ap = pysptk.sptk.mgc2sp(x.astype(np.float64), order=hp.n_bap - 1,
                            alpha=hp.mcep_alpha, gamma=0.0, fftlen=hp.n_fft)
    ap = np.square(ap / int16_max)
    for i in range(len(lf0)):
        ap[i] = np.where(lf0[i] == 0, np.zeros(ap.shape[1]), ap[i])
    return ap

def plot_alignment(alignment, gs, dir=hp.logdir):
    """Plots the alignment.

    Args:
      alignment: A numpy array with shape of (encoder_steps, decoder_steps)
      gs: (int) global step.
      dir: Output path.
    """
    if not os.path.exists(dir): os.mkdir(dir)

    fig, ax = plt.subplots()
    im = ax.imshow(alignment)

    fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.savefig('{}/alignment_{}.png'.format(dir, gs), format='png')
    plt.close(fig)

def guided_attention(g=0.2):
    '''Guided attention. Refer to page 3 on the paper.'''
    W = np.zeros((hp.max_N, hp.max_T), dtype=np.float32)
    for n_pos in range(W.shape[0]):
        for t_pos in range(W.shape[1]):
            W[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(hp.max_T) - n_pos / float(hp.max_N)) ** 2 / (2 * g * g))
    return W

def learning_rate_decay(init_lr, global_step, warmup_steps = 4000.0):
    '''Noam scheme from tensor2tensor'''
    step = tf.to_float(global_step + 1)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)

def load_features(fpath, text_len):
    '''Read the wave file in `fpath`
    and extracts world vocoder features'''

    fname = os.path.basename(fpath)
    wav, _ = sf.read(fpath)

    f0, sp, ap = vocoder.wav2world(wav, hp.sr, hp.n_fft)

    # Marginal padding for reduction shape sync.
    # num_paddings = hp.r - (len(f0) % hp.r) if len(f0) % hp.r != 0 else 0
    # mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    # mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")

    if len(f0) > hp.max_T or text_len > hp.max_N:
        return None

    # Normalization and reduction
    lf0 = f0_norm(f0)[::hp.r]
    mgc = sp_norm(sp)[::hp.r, :]
    bap = ap_norm(ap)[::hp.r, :]

    np.save("lf0/{}".format(fname.replace("wav", "npy")), lf0)
    np.save("mgc/{}".format(fname.replace("wav", "npy")), mgc)
    np.save("bap/{}".format(fname.replace("wav", "npy")), bap)
    return (fname, lf0, mgc, bap)
