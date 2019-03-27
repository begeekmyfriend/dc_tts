# -*- coding: utf-8 -*-
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''
from __future__ import print_function, division

import pysptk
import pyworld
import soundfile as sf
import numpy as np
import os
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from hyperparams import Hyperparams as hp
import tensorflow as tf

def save_wav(wav, path):
    sf.write(path, wav, hp.sr)

def world_synthesize(feature):
    mgc_idx = 0
    lf0_idx = mgc_idx + hp.n_mgc
    vuv_idx = lf0_idx + hp.n_lf0
    bap_idx = vuv_idx + hp.n_vuv

    mgc = feature[:, mgc_idx : mgc_idx + hp.n_mgc]
    lf0 = feature[:, lf0_idx : lf0_idx + hp.n_lf0]
    vuv = feature[:, vuv_idx : vuv_idx + hp.n_vuv]
    bap = feature[:, bap_idx : bap_idx + hp.n_bap]

    fs = hp.sr
    alpha = pysptk.util.mcepalpha(fs)
    fftlen = fftlen = pyworld.get_cheaptrick_fft_size(fs)

    spectrogram = pysptk.mc2sp(mgc, fftlen=fftlen, alpha=alpha)
    aperiodicity = pyworld.decode_aperiodicity(bap.astype(np.float64), fs, fftlen)
    f0 = lf0.copy()
    f0[vuv < 0.5] = 0
    f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])

    return pyworld.synthesize(f0.flatten().astype(np.float64),
                              spectrogram.astype(np.float64),
                              aperiodicity.astype(np.float64),
                              fs, hp.frame_period)

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
    N = hp.max_N
    T = hp.max_T // hp.r
    W = np.zeros((N, T), dtype=np.float32)
    for n_pos in range(W.shape[0]):
        for t_pos in range(W.shape[1]):
            W[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(T) - n_pos / float(N)) ** 2 / (2 * g * g))
    return W

def learning_rate_decay(init_lr, global_step, warmup_steps = 4000.0):
    '''Noam scheme from tensor2tensor'''
    step = tf.to_float(global_step + 1)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)

def load_features(fpath, text_len):
    '''Read the wave file in `fpath`
    and extracts world vocoder features'''
    wav, fs = sf.read(fpath)
    if hp.use_harvest:
        f0, timeaxis = pyworld.harvest(wav, fs, frame_period=hp.frame_period, f0_floor=hp.f0_floor, f0_ceil=hp.f0_ceil)
    else:
        f0, timeaxis = pyworld.dio(wav, fs, frame_period=hp.frame_period, f0_floor=hp.f0_floor, f0_ceil=hp.f0_ceil)
        f0 = pyworld.stonemask(wav, f0, timeaxis, fs)

    if len(f0) > hp.max_T or text_len > hp.max_N:
        return None

    spectrogram = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    aperiodicity = pyworld.d4c(wav, f0, timeaxis, fs)
    bap = pyworld.code_aperiodicity(aperiodicity, fs)
    alpha = pysptk.util.mcepalpha(fs)
    mgc = pysptk.sp2mc(spectrogram, order=hp.n_mgc - 1, alpha=alpha)
    f0 = f0[:, None]
    lf0 = f0.copy()
    nonzero_indices = np.nonzero(f0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    if hp.use_harvest:
        # https://github.com/mmorise/World/issues/35#issuecomment-306521887
        vuv = (aperiodicity[:, 0] < 0.5).astype(np.float32)[:, None]
    else:
        vuv = (lf0 != 0).astype(np.float32)
    # lf0 = P.interp1d(lf0, kind=hp.f0_interpolation_kind)

    # Parameter trajectory smoothing
    # if hp.mod_spec_smoothing:
    #     hop_length = int(fs * (hp.frame_period * 0.001))
    #     modfs = fs / hop_length
    #     mgc = P.modspec_smoothing(
    #     mgc, modfs, cutoff=hp.mod_spec_smoothing_cutoff)

    # mgc = P.delta_features(mgc, hp.windows)
    # lf0 = P.delta_features(lf0, hp.windows)
    # bap = P.delta_features(bap, hp.windows)

    features = np.hstack((mgc, lf0, vuv, bap))

    # Normalization and reduction
    features = features.astype(np.float32)[::hp.r, :]

    fname = os.path.basename(fpath)
    np.save("features/{}".format(fname.replace("wav", "npy")), features)

    return (fname, features)
