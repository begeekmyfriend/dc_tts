# -*- coding: utf-8 -*-
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from utils import *
from hyperparams import Hyperparams as hp
import os
from data_load import load_data
import numpy as np

# Load data
if not os.path.exists("features"): os.mkdir("features")

fpaths, text_lens, _ = load_data('prepro')
futures = []
executor = ProcessPoolExecutor(max_workers=16)

for i, fpath in enumerate(fpaths):
    text_len = text_lens[i]
    futures.append(executor.submit(partial(load_features, fpath, text_len)))

results = [future.result() for future in tqdm(futures) if future.result() is not None]
n_frames = sum([len(res[1]) for res in results])
timesteps = n_frames * hp.frame_period
hours = timesteps / 3600
print('Write ({} clips({:.2f} hours)'.format(len(results), hours))
print('Max input length (text chars): {}'.format(max(text_lens)))
print('Max mel frames length: {}'.format(max(len(res[1]) for res in results)))
print('Max audio timesteps length: {}'.format(max(int(len(res[1]) * hp.frame_period) for res in results)))
