# -*- coding: utf-8 -*-
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from utils import load_spectrograms
from hyperparams import Hyperparams as hp
import os
from data_load import load_data
import numpy as np

# Load data
if not os.path.exists("mels"): os.mkdir("mels")
if not os.path.exists("mags"): os.mkdir("mags")

fpaths, text_lens, _ = load_data('prepro') # list
futures = []
executor = ProcessPoolExecutor(max_workers=16)

for i, fpath in enumerate(fpaths):
    text_len = text_lens[i]
    futures.append(executor.submit(partial(load_spectrograms, fpath, text_len)))

results = [future.result() for future in tqdm(futures) if future.result() is not None]
mel_frames = sum([int(res[1].shape[0]) for res in results])
timesteps = mel_frames * hp.frame_shift
hours = timesteps / hp.sr / 3600
print('Write ({} clips({:.2f} hours)'.format(len(results), hours))
print('Max input length (text chars): {}'.format(max(text_lens)))
print('Max mel frames length: {}'.format(max(int(res[1].shape[0]) for res in results)))
print('Max audio timesteps length: {}'.format(max(int(res[1].shape[0] * hp.frame_shift) for res in results)))
