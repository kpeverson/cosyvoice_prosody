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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm

# Thread-local HDF5 handles so each thread opens its own file descriptor.
_thread_local = threading.local()


def _get_h5():
    if not hasattr(_thread_local, 'h5_file'):
        import h5py
        _thread_local.h5_file = h5py.File(args.hdf5_file, 'r')
    return _thread_local.h5_file


def single_job(utt):
    try:
        if args.hdf5_file:
            arr = np.array(_get_h5()[utt][:], dtype=np.float32)
            audio = torch.from_numpy(arr)
            if arr.ndim == 1:
                audio = audio.unsqueeze(0)  # [1, T]
            sample_rate = args.sample_rate
        else:
            audio, sample_rate = torchaudio.load(utt2wav[utt])
    except KeyError:
        return utt, None
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    feat = kaldi.fbank(audio,
                       num_mel_bins=80,
                       dither=0,
                       sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
    return utt, embedding


def main(args):
    utt_list = list(utt2wav.keys())
    all_task = [executor.submit(single_job, utt) for utt in utt_list]
    utt2embedding, spk2embedding = {}, {}
    skipped = 0
    for future in tqdm(as_completed(all_task), total=len(all_task)):
        utt, embedding = future.result()
        if embedding is None:
            skipped += 1
            continue
        utt2embedding[utt] = embedding
        spk = utt2spk[utt]
        if spk not in spk2embedding:
            spk2embedding[spk] = []
        spk2embedding[spk].append(embedding)
    if skipped:
        print(f'Warning: {skipped} utterances skipped (not found in HDF5)')
    for k, v in spk2embedding.items():
        spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()
    torch.save(utt2embedding, "{}/utt2embedding.pt".format(args.dir))
    torch.save(spk2embedding, "{}/spk2embedding.pt".format(args.dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--onnx_path", type=str)
    parser.add_argument("--num_thread", type=int, default=8)
    parser.add_argument("--hdf5_file", type=str, default=None,
                        help="HDF5 file containing waveforms keyed by utt_id. "
                             "When set, wav.scp paths are ignored for audio loading.")
    parser.add_argument("--sample_rate", type=int, default=24000,
                        help="Sample rate of waveforms stored in the HDF5 file.")
    args = parser.parse_args()

    utt2wav, utt2spk = {}, {}
    with open('{}/wav.scp'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    with open('{}/utt2spk'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2spk[l[0]] = l[1]

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)
    executor = ThreadPoolExecutor(max_workers=args.num_thread)

    main(args)
