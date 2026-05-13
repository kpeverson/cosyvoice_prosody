#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import io
import logging
import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing
import time
import torch


def job(utt_list, parquet_file, utt2parquet_file, spk2parquet_file):
    start_time = time.time()
    data_list = []
    loaded_utts = []
    if args.hdf5_file:
        import h5py
        import torch
        import torchaudio
        h5 = h5py.File(args.hdf5_file, 'r')
    for utt in tqdm(utt_list):
        try:
            if args.hdf5_file:
                arr = np.array(h5[utt][:], dtype=np.float32)
                tensor = torch.from_numpy(arr)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)  # [1, T]
                buf = io.BytesIO()
                torchaudio.save(buf, tensor, sample_rate=args.sample_rate, format='wav')
                data = buf.getvalue()
            else:
                data = open(utt2wav[utt], 'rb').read()
        except KeyError:
            logging.warning('utt %s not found in HDF5 file, skipping', utt)
            continue
        data_list.append(data)
        loaded_utts.append(utt)
    if args.hdf5_file:
        h5.close()
    utt_list = loaded_utts
    spk_list = [utt2spk[utt] for utt in utt_list]

    # 保存到parquet,utt2parquet_file,spk2parquet_file
    df = pd.DataFrame()
    df['utt'] = utt_list
    df['audio_data'] = data_list
    df['wav'] = [utt2wav[utt] for utt in utt_list]
    df['text'] = [utt2text[utt] for utt in utt_list]
    df['spk'] = [utt2spk[utt] for utt in utt_list]
    if utt2embedding is not None:
        df['utt_embedding'] = [utt2embedding[utt] for utt in utt_list]
    if spk2embedding is not None:
        df['spk_embedding'] = [spk2embedding[utt2spk[utt]] for utt in utt_list]
    if utt2speech_token is not None:
        df['speech_token'] = [utt2speech_token[utt] for utt in utt_list]
    if utt2instruct is not None:
        df['instruct'] = [utt2instruct[utt] for utt in utt_list]
    if utt2prosody_token is not None:
        df['prosody_token'] = [utt2prosody_token.get(utt, []) for utt in utt_list]
    if args.dpo:
        df['reject_speech_token'] = [utt2reject_speech_token.get(utt, None) for utt in utt_list]
    df.to_parquet(parquet_file)
    with open(utt2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in utt_list}, f, ensure_ascii=False, indent=2)
    with open(spk2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in list(set(spk_list))}, f, ensure_ascii=False, indent=2)
    logging.info('spend time {}'.format(time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_utts_per_parquet',
                        type=int,
                        default=1000,
                        help='num utts per parquet')
    parser.add_argument('--num_processes',
                        type=int,
                        default=1,
                        help='num processes for make parquets')
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    parser.add_argument('--dpo',
                        action='store_true',
                        default=False,
                        help='Use Direct Preference Optimization')
    parser.add_argument('--hdf5_file', type=str, default=None,
                        help='HDF5 file containing waveforms keyed by utt_id. '
                             'When set, wav.scp paths are ignored for audio loading.')
    parser.add_argument('--sample_rate', type=int, default=24000,
                        help='Sample rate of waveforms stored in the HDF5 file.')
    parser.add_argument('--nshard', type=int, default=1,
                        help='Total number of shards for multi-node parallelism.')
    parser.add_argument('--shard_rank', type=int, default=0,
                        help='0-indexed rank of this shard. Each node runs with a different value.')
    args = parser.parse_args()

    utt2wav, utt2text, utt2spk = {}, {}, {}
    with open('{}/wav.scp'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    with open('{}/text'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2text[l[0]] = ' '.join(l[1:])
    with open('{}/utt2spk'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2spk[l[0]] = l[1]
    if os.path.exists('{}/instruct'.format(args.src_dir)):
        utt2instruct = {}
        with open('{}/instruct'.format(args.src_dir)) as f:
            for l in f:
                l = l.replace('\n', '').split()
                utt2instruct[l[0]] = ' '.join(l[1:])
    else:
        utt2instruct = None
    utt2embedding = torch.load('{}/utt2embedding.pt'.format(args.src_dir)) if os.path.exists('{}/utt2embedding.pt'.format(args.src_dir)) else None
    spk2embedding = torch.load('{}/spk2embedding.pt'.format(args.src_dir)) if os.path.exists('{}/spk2embedding.pt'.format(args.src_dir)) else None
    utt2speech_token = torch.load('{}/utt2speech_token.pt'.format(args.src_dir)) if os.path.exists('{}/utt2speech_token.pt'.format(args.src_dir)) else None
    utt2prosody_token = torch.load('{}/utt2prosody_token.pt'.format(args.src_dir)) if os.path.exists('{}/utt2prosody_token.pt'.format(args.src_dir)) else None
    if args.dpo:
        utt2reject_speech_token = torch.load('{}_reject/utt2speech_token.pt'.format(args.src_dir)) if os.path.exists('{}_reject/utt2speech_token.pt'.format(args.src_dir)) else {}
    utts = list(utt2wav.keys())
    # slice this node's portion
    utts = utts[args.shard_rank::args.nshard]
    logging.info('shard %d/%d: %d utterances', args.shard_rank, args.nshard, len(utts))

    # Using process pool to speedup
    pool = multiprocessing.Pool(processes=args.num_processes)
    parquet_list, utt2parquet_list, spk2parquet_list = [], [], []
    for i, j in enumerate(range(0, len(utts), args.num_utts_per_parquet)):
        parquet_file = os.path.join(args.des_dir, 'parquet_{:03d}_{:06d}.tar'.format(args.shard_rank, i))
        utt2parquet_file = os.path.join(args.des_dir, 'utt2parquet_{:03d}_{:06d}.json'.format(args.shard_rank, i))
        spk2parquet_file = os.path.join(args.des_dir, 'spk2parquet_{:03d}_{:06d}.json'.format(args.shard_rank, i))
        parquet_list.append(parquet_file)
        utt2parquet_list.append(utt2parquet_file)
        spk2parquet_list.append(spk2parquet_file)
        pool.apply_async(job, (utts[j: j + args.num_utts_per_parquet], parquet_file, utt2parquet_file, spk2parquet_file),
                         error_callback=lambda e: logging.error('Parquet job failed: %s', e))
    pool.close()
    pool.join()

    suffix = '' if args.nshard == 1 else '.{:03d}'.format(args.shard_rank)
    with open('{}/data{}.list'.format(args.des_dir, suffix), 'w', encoding='utf8') as f1, \
            open('{}/utt2data{}.list'.format(args.des_dir, suffix), 'w', encoding='utf8') as f2, \
            open('{}/spk2data{}.list'.format(args.des_dir, suffix), 'w', encoding='utf8') as f3:
        for name in parquet_list:
            f1.write(name + '\n')
        for name in utt2parquet_list:
            f2.write(name + '\n')
        for name in spk2parquet_list:
            f3.write(name + '\n')
